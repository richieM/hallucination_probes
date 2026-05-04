"""Generate synthetic conversations for authority-projection probing.

Run:
    uv run python -m auth_projection.generate --n_per_cell 1   # tiny pilot
    uv run python -m auth_projection.generate --n_per_cell 20  # full ~600-conv batch
"""

import asyncio
import logging
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
    CommunicationStyle,
    Conversation,
    ConversationalGoal,
    GenerationOutput,
    GenerationSeed,
    LifeContext,
    Persona,
    PriorAIExperience,
    Tier,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
assert os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY is not set"

PROMPT_PATH = Path(__file__).parent / "prompts" / "generation.prompt"
PROMPT_TEMPLATE = PROMPT_PATH.read_text().strip()
DEFAULT_OUTPUT = Path(__file__).parent / "data" / "generated.jsonl"

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
    lexical_twin_fraction: float = 0.3
    n_turns_choices: Tuple[int, ...] = (3, 4, 5)

    output_path: Path = DEFAULT_OUTPUT
    seed: int = 42

    datetime_str: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    safetytooling_cache_dir: Union[str, Path] = Path.home() / ".safetytooling_cache"
    save_path: Optional[Path] = None
    log_to_file: bool = False

    def __post_init__(self):
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)


def build_seeds(cfg: GenerateConfig) -> List[GenerationSeed]:
    rng = random.Random(cfg.seed)
    seeds: List[GenerationSeed] = []
    for topic in TOPICS:
        for tier in TIERS:
            n_lex_twin = (
                int(round(cfg.n_per_cell * cfg.lexical_twin_fraction))
                if tier in ("none", "strongly")
                else 0
            )
            for i in range(cfg.n_per_cell):
                persona = Persona(
                    age_bucket=rng.choice(AGE_BUCKETS),
                    communication_style=rng.choice(COMM_STYLES),
                    prior_ai_experience=rng.choice(AI_EXP),
                    life_context=rng.choice(LIFE_CTX),
                    conversational_goal=rng.choice(GOALS),
                )
                seeds.append(
                    GenerationSeed.make(
                        topic=topic,
                        persona=persona,
                        target_tier=tier,
                        n_turns=rng.choice(cfg.n_turns_choices),
                        uses_lexical_twin=(i < n_lex_twin),
                        salt=str(i),
                    )
                )
    return seeds


def load_existing_seed_ids(path: Path) -> set:
    if not path.exists():
        return set()
    return {row["seed_id"] for row in load_jsonl(path) if "seed_id" in row}


def format_prompt(seed: GenerationSeed) -> str:
    return (
        PROMPT_TEMPLATE.replace("{topic}", seed.topic)
        .replace("{persona_description}", seed.persona.to_prompt_description())
        .replace("{target_tier}", seed.target_tier)
        .replace("{n_turns}", str(seed.n_turns))
        .replace("{uses_lexical_twin}", str(seed.uses_lexical_twin).lower())
    )


async def generate_one(
    seed: GenerationSeed,
    cfg: GenerateConfig,
    inference_api: InferenceAPI,
) -> Optional[Conversation]:
    user_prompt = format_prompt(seed)
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
            uses_lexical_twin=seed.uses_lexical_twin,
            conversation=gen_output.conversation,
        )
        save_jsonl(conv.model_dump(), cfg.output_path, append=True)
        return conv
    except Exception as e:
        logger.error(f"seed {seed.seed_id}: generation failed: {e}")
        return None


async def main(cfg: GenerateConfig):
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

    tasks = [generate_one(s, cfg, inference_api) for s in todo]
    results = await tqdm_asyncio.gather(*tasks, desc="Generating")
    successful = sum(1 for r in results if r is not None)
    logger.info(f"Generated {successful}/{len(todo)} conversations -> {cfg.output_path}")


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(GenerateConfig, dest="config")
    args = parser.parse_args()
    asyncio.run(main(args.config))
