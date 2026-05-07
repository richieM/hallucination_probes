"""Generate synthetic conversations for authority-projection probing.

Run:
    uv run python -m auth_projection.generate --n_per_cell 1   # tiny pilot
    uv run python -m auth_projection.generate --n_per_cell 20  # full ~600-conv batch
"""

import asyncio
import logging
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import simple_parsing
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, LLMResponse, MessageRole, Prompt
from safetytooling.utils.experiment_utils import ExperimentConfigBase

from utils.file_utils import load_jsonl, save_jsonl
from utils.parsing import parse_and_validate_json

from .data_models import (
    AgeBucket,
    ArcShape,
    CommunicationStyle,
    Conversation,
    ConversationalGoal,
    GenerationOutput,
    GenerationSeed,
    LexicalTwinKind,
    LifeContext,
    Persona,
    PriorAIExperience,
    Tier,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
assert os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY is not set"

DEFAULT_PROMPT_PATH = Path(__file__).parent / "prompts" / "generation_v1.prompt"
DEFAULT_OUTPUT = Path(__file__).parent / "data" / "generated.jsonl"

ARC_DIST: dict = {
    "none": {"stable": 0.7, "mixed": 0.3},
    "somewhat": {"stable": 0.4, "escalating": 0.3, "de_escalating": 0.15, "mixed": 0.15},
    "strongly": {"stable": 0.35, "escalating": 0.35, "de_escalating": 0.15, "mixed": 0.15},
}

LEX_DIST: dict = {
    "none": {"match": 0.7, "submission_voice": 0.3},
    "somewhat": {"match": 0.6, "peer_voice": 0.2, "submission_voice": 0.2},
    "strongly": {"match": 0.7, "peer_voice": 0.3},
}

TOPICS: List[str] = [
    "career decisions (job offers, switching fields, promotions)",
    "financial decisions (budgeting, large purchases, investing basics)",
    "relationship advice (interpersonal conflict, communication)",
    "cooking and recipes",
    "travel planning",
    "programming or technical problems",
    "philosophy or ethics dilemmas",
    "writing or creative projects",
    "personal goals, habits, or productivity",
    "health and fitness routines",
]
TIERS: List[Tier] = ["none", "somewhat", "strongly"]
AGE_BUCKETS: List[AgeBucket] = ["20s", "30s", "40s", "50s+"]
COMM_STYLES: List[CommunicationStyle] = ["terse", "verbose", "casual", "formal"]
AI_EXP: List[PriorAIExperience] = ["none", "casual", "heavy"]
LIFE_CTX: List[LifeContext] = ["stable", "mild_stress", "major_transition"]
GOALS: List[ConversationalGoal] = [
    "seeking_information",
    "brainstorming",
    "venting",
    "validation_seeking",
    "pressure_testing",
]


@dataclass
class GenerateConfig(ExperimentConfigBase):
    model_id: str = "claude-sonnet-4-6"
    temperature: float = 0.9
    max_tokens: int = 2048
    max_concurrent_tasks: int = 10

    n_per_cell: int = 2
    n_per_cell_none: Optional[int] = None       # if set, overrides n_per_cell for tier=none
    n_per_cell_somewhat: Optional[int] = None   # if set, overrides n_per_cell for tier=somewhat
    n_per_cell_strongly: Optional[int] = None   # if set, overrides n_per_cell for tier=strongly
    n_turns_choices: Tuple[int, ...] = (3, 4, 5)
    limit_topics: Optional[int] = None  # if set, randomly subset topics to this many (pilot mode)

    prompt_path: Path = DEFAULT_PROMPT_PATH
    output_path: Path = DEFAULT_OUTPUT
    output_dir: Path = DEFAULT_OUTPUT.parent
    seed: int = 42

    datetime_str: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    safetytooling_cache_dir: Union[str, Path] = Path.home() / ".safetytooling_cache"
    save_path: Optional[Path] = None
    log_to_file: bool = False

    def __post_init__(self):
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)


def _stratified_from_dist(dist: dict, n: int, rng: random.Random) -> List[str]:
    """Return n samples from a {value: weight} dist whose mix approximates the dist.

    Uses largest-remainder allocation so no bucket gets overshot and trimmed away —
    important for small n where rounding losses can drop whole categories.
    """
    out: list = []
    remainders = []
    for v, w in dist.items():
        whole = int(w * n)  # floor
        out.extend([v] * whole)
        remainders.append((w * n - whole, v))
    deficit = n - len(out)
    if deficit > 0:
        keys = [v for _, v in remainders]
        weights = [r for r, _ in remainders]
        for _ in range(deficit):
            out.append(rng.choices(keys, weights=weights)[0])
    rng.shuffle(out)
    return out


def stratified_arc_shapes(tier: Tier, n: int, rng: random.Random) -> List[ArcShape]:
    return _stratified_from_dist(ARC_DIST[tier], n, rng)  # type: ignore[return-value]


def stratified_lexical_kinds(tier: Tier, n: int, rng: random.Random) -> List[LexicalTwinKind]:
    return _stratified_from_dist(LEX_DIST[tier], n, rng)  # type: ignore[return-value]


def build_seeds(cfg: GenerateConfig) -> List[GenerationSeed]:
    rng = random.Random(cfg.seed)
    topics = list(TOPICS)
    if cfg.limit_topics is not None and cfg.limit_topics < len(topics):
        topics = rng.sample(topics, cfg.limit_topics)

    per_cell_by_tier: dict = {
        "none": cfg.n_per_cell_none if cfg.n_per_cell_none is not None else cfg.n_per_cell,
        "somewhat": cfg.n_per_cell_somewhat if cfg.n_per_cell_somewhat is not None else cfg.n_per_cell,
        "strongly": cfg.n_per_cell_strongly if cfg.n_per_cell_strongly is not None else cfg.n_per_cell,
    }

    arc_shapes_by_tier: dict = {
        t: stratified_arc_shapes(t, len(topics) * per_cell_by_tier[t], rng) for t in TIERS
    }
    lex_kinds_by_tier: dict = {
        t: stratified_lexical_kinds(t, len(topics) * per_cell_by_tier[t], rng) for t in TIERS
    }
    cursor: dict = {t: 0 for t in TIERS}

    seeds: List[GenerationSeed] = []
    for topic in topics:
        for tier in TIERS:
            for _ in range(per_cell_by_tier[tier]):
                persona = Persona(
                    age_bucket=rng.choice(AGE_BUCKETS),
                    communication_style=rng.choice(COMM_STYLES),
                    prior_ai_experience=rng.choice(AI_EXP),
                    life_context=rng.choice(LIFE_CTX),
                    conversational_goal=rng.choice(GOALS),
                )
                idx = cursor[tier]
                cursor[tier] += 1
                seeds.append(
                    GenerationSeed.make(
                        topic=topic,
                        persona=persona,
                        target_tier=tier,
                        arc_shape=arc_shapes_by_tier[tier][idx],
                        lexical_twin_kind=lex_kinds_by_tier[tier][idx],
                        n_turns=rng.choice(cfg.n_turns_choices),
                        salt=str(idx),
                    )
                )
    return seeds


def load_existing_seed_ids(path: Path) -> set:
    if not path.exists():
        return set()
    return {row["seed_id"] for row in load_jsonl(path) if "seed_id" in row}


def format_prompt(seed: GenerationSeed, template: str) -> str:
    return (
        template.replace("{topic}", seed.topic)
        .replace("{persona_description}", seed.persona.to_prompt_description())
        .replace("{target_tier}", seed.target_tier)
        .replace("{arc_shape}", seed.arc_shape)
        .replace("{lexical_twin_kind}", seed.lexical_twin_kind)
        .replace("{n_turns}", str(seed.n_turns))
    )


async def generate_one(
    seed: GenerationSeed,
    cfg: GenerateConfig,
    inference_api: InferenceAPI,
    prompt_template: str,
) -> Optional[Conversation]:
    user_prompt = format_prompt(seed, prompt_template)
    try:
        response: List[LLMResponse] = await inference_api(
            model_id=cfg.model_id,
            prompt=Prompt(messages=[ChatMessage(role=MessageRole.user, content=user_prompt)]),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        gen_output: GenerationOutput = parse_and_validate_json(
            response[0].completion, GenerationOutput, allow_partial=True
        )

        n_user = sum(1 for t in gen_output.conversation if t.role == "user")
        if n_user != seed.n_turns:
            logger.warning(
                f"seed {seed.seed_id}: requested {seed.n_turns} user turns, got {n_user}"
            )

        conv = Conversation(
            seed_id=seed.seed_id,
            topic=seed.topic,
            persona=seed.persona,
            target_tier=seed.target_tier,
            arc_shape=seed.arc_shape,
            lexical_twin_kind=seed.lexical_twin_kind,
            conversation=gen_output.conversation,
        )
        save_jsonl(conv.model_dump(), cfg.output_path, append=True)
        return conv
    except Exception as e:
        logger.error(f"seed {seed.seed_id}: generation failed: {e}")
        return None


async def main(cfg: GenerateConfig):
    prompt_template = Path(cfg.prompt_path).read_text().strip()
    seeds = build_seeds(cfg)
    existing = load_existing_seed_ids(cfg.output_path)
    todo = [s for s in seeds if s.seed_id not in existing]
    logger.info(
        f"Total seeds: {len(seeds)} | already done: {len(existing)} | to generate: {len(todo)}"
    )
    if not todo:
        return

    inference_api = InferenceAPI(
        cache_dir=cfg.safetytooling_cache_dir,
        anthropic_num_threads=cfg.max_concurrent_tasks,
    )

    tasks = [generate_one(s, cfg, inference_api, prompt_template) for s in todo]
    results = await tqdm_asyncio.gather(*tasks, desc="Generating")
    successful = sum(1 for r in results if r is not None)
    logger.info(f"Generated {successful}/{len(todo)} conversations -> {cfg.output_path}")


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(GenerateConfig, dest="config")
    args = parser.parse_args()
    asyncio.run(main(args.config))
