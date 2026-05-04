"""Generate counterfactual eval-set conversations (paraphrases or minimal-edit pairs).

Run:
    uv run python -m auth_projection.generate_counterfactuals --mode paraphrase
    uv run python -m auth_projection.generate_counterfactuals --mode minimal_edit
"""

import asyncio
import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import simple_parsing
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, LLMResponse, MessageRole, Prompt
from safetytooling.utils.experiment_utils import ExperimentConfigBase

from utils.file_utils import load_jsonl, save_jsonl
from utils.parsing import parse_and_validate_json

from .data_models import Conversation, GenerationOutput, Tier
from .label import format_conversation_for_labeler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
assert os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY is not set"


PARA_PROMPT_PATH = Path(__file__).parent / "prompts" / "paraphrase.prompt"
MINEDIT_PROMPT_PATH = Path(__file__).parent / "prompts" / "minimal_edit.prompt"
DEFAULT_INPUT = Path(__file__).parent / "data" / "labeled.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "data"


def flip_target(tier: Tier) -> Tier:
    """Pick a flipped tier for minimal-edit. Sample uniformly from the other two."""
    others = [t for t in ("none", "somewhat", "strongly") if t != tier]
    return random.choice(others)  # caller seeds rng


@dataclass
class CounterfactualConfig(ExperimentConfigBase):
    mode: str = "paraphrase"  # "paraphrase" | "minimal_edit"
    model_id: str = "claude-sonnet-4-6"
    temperature: float = 0.7
    max_tokens: int = 2048
    max_concurrent_tasks: int = 10

    input_path: Path = DEFAULT_INPUT
    output_path: Optional[Path] = None
    output_dir: Path = DEFAULT_OUTPUT_DIR

    # Pick a held-out subset of the labeled corpus to counterfactualize.
    # Same train/test split as train_probe.py: 20% by seed, seed=42.
    test_frac: float = 0.2
    split_seed: int = 42

    # Cap counterfactual count for cost control. None = all held-out.
    max_n: Optional[int] = 100

    # For minimal_edit: rng seed for choosing flip targets
    flip_seed: int = 7

    datetime_str: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    safetytooling_cache_dir: Union[str, Path] = Path.home() / ".safetytooling_cache"
    save_path: Optional[Path] = None
    log_to_file: bool = False

    def __post_init__(self):
        if self.output_path is None:
            self.output_path = self.output_dir / f"counterfactual_{self.mode}.jsonl"
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)


def select_held_out(rows: List[dict], test_frac: float, split_seed: int) -> List[dict]:
    """Pick the same held-out seed_ids as train_probe's test split."""
    import numpy as np

    seed_ids = sorted({r["seed_id"] for r in rows})
    rng = np.random.RandomState(split_seed)
    rng.shuffle(seed_ids)
    n_test = max(1, int(len(seed_ids) * test_frac))
    test_set = set(seed_ids[:n_test])
    return [r for r in rows if r["seed_id"] in test_set]


def format_paraphrase_prompt(conv: Conversation) -> str:
    template = PARA_PROMPT_PATH.read_text().strip()
    return template.replace("{target_tier}", conv.target_tier).replace(
        "{conversation_formatted}", format_conversation_for_labeler(conv)
    )


def format_minimal_edit_prompt(conv: Conversation, target_flip: str) -> str:
    template = MINEDIT_PROMPT_PATH.read_text().strip()
    return (
        template.replace("{target_tier}", conv.target_tier)
        .replace("{target_flip}", target_flip)
        .replace("{conversation_formatted}", format_conversation_for_labeler(conv))
    )


async def cf_one(
    conv: Conversation,
    cfg: CounterfactualConfig,
    inference_api: InferenceAPI,
    target_flip: Optional[str] = None,
) -> Optional[Conversation]:
    if cfg.mode == "paraphrase":
        prompt_text = format_paraphrase_prompt(conv)
        new_target_tier = conv.target_tier
        marker = "para"
    elif cfg.mode == "minimal_edit":
        if target_flip is None:
            raise ValueError("minimal_edit needs target_flip")
        prompt_text = format_minimal_edit_prompt(conv, target_flip)
        new_target_tier = target_flip
        marker = f"minedit_to_{target_flip}"
    else:
        raise ValueError(f"unknown mode {cfg.mode}")

    try:
        response: List[LLMResponse] = await inference_api(
            model_id=cfg.model_id,
            prompt=Prompt(messages=[ChatMessage(role=MessageRole.user, content=prompt_text)]),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        gen_output: GenerationOutput = parse_and_validate_json(
            response[0].completion, GenerationOutput, allow_partial=True
        )
        new_conv = Conversation(
            seed_id=f"{conv.seed_id}__{marker}",
            topic=conv.topic,
            persona=conv.persona,
            target_tier=new_target_tier,
            uses_lexical_twin=conv.uses_lexical_twin,
            conversation=gen_output.conversation,
        )
        # Tag with parent + cf type for downstream analysis
        new_conv.__pydantic_extra__ = {
            "parent_seed_id": conv.seed_id,
            "cf_mode": cfg.mode,
            "original_target_tier": conv.target_tier,
        }
        save_jsonl(new_conv.model_dump(), cfg.output_path, append=True)
        return new_conv
    except Exception as e:
        logger.error(f"seed {conv.seed_id} ({cfg.mode}): failed: {e}")
        return None


async def main(cfg: CounterfactualConfig):
    rows = load_jsonl(cfg.input_path)
    held_out = select_held_out(rows, cfg.test_frac, cfg.split_seed)
    logger.info(f"Held-out pool: {len(held_out)} convs")

    if cfg.max_n and len(held_out) > cfg.max_n:
        held_out = held_out[: cfg.max_n]
    convs = [Conversation.model_validate(r) for r in held_out]

    # Resume: skip already-generated children
    existing_children = set()
    if cfg.output_path.exists():
        existing_children = {r["seed_id"] for r in load_jsonl(cfg.output_path)}
    convs_todo = []
    for c in convs:
        marker_prefix = f"{c.seed_id}__"
        if any(child.startswith(marker_prefix) for child in existing_children):
            continue
        convs_todo.append(c)
    logger.info(f"To {cfg.mode}: {len(convs_todo)} convs (skipping {len(convs) - len(convs_todo)})")

    if not convs_todo:
        return

    inference_api = InferenceAPI(
        cache_dir=cfg.safetytooling_cache_dir,
        anthropic_num_threads=cfg.max_concurrent_tasks,
    )

    rng = random.Random(cfg.flip_seed)
    tasks = []
    for c in convs_todo:
        if cfg.mode == "minimal_edit":
            target_flip = flip_target(c.target_tier)
        else:
            target_flip = None
        tasks.append(cf_one(c, cfg, inference_api, target_flip=target_flip))

    results = await tqdm_asyncio.gather(*tasks, desc=f"Counterfactual ({cfg.mode})")
    successful = sum(1 for r in results if r is not None)
    logger.info(f"Wrote {successful}/{len(tasks)} counterfactuals -> {cfg.output_path}")


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(CounterfactualConfig, dest="config")
    args = parser.parse_args()
    asyncio.run(main(args.config))
