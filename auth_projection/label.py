"""Label generated conversations for authority projection.

Run:
    uv run python -m auth_projection.label
    uv run python -m auth_projection.label --input_path auth_projection/data/generated.jsonl --output_path auth_projection/data/labeled.jsonl
"""

import asyncio
import logging
import os
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

from .data_models import Conversation, LabelingOutput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
assert os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY is not set"

PROMPT_PATH = Path(__file__).parent / "prompts" / "labeler.prompt"
PROMPT_TEMPLATE = PROMPT_PATH.read_text().strip()
DEFAULT_INPUT = Path(__file__).parent / "data" / "generated.jsonl"
DEFAULT_OUTPUT = Path(__file__).parent / "data" / "labeled.jsonl"


@dataclass
class LabelConfig(ExperimentConfigBase):
    model_id: str = "claude-sonnet-4-6"
    temperature: float = 0.0
    max_tokens: int = 2048
    max_concurrent_tasks: int = 10

    input_path: Path = DEFAULT_INPUT
    output_path: Path = DEFAULT_OUTPUT
    output_dir: Path = DEFAULT_OUTPUT.parent

    datetime_str: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    safetytooling_cache_dir: Union[str, Path] = Path.home() / ".safetytooling_cache"
    save_path: Optional[Path] = None
    log_to_file: bool = False

    def __post_init__(self):
        if isinstance(self.input_path, str):
            self.input_path = Path(self.input_path)
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)


def format_conversation_for_labeler(conv: Conversation) -> str:
    """Render a conversation with explicit user-turn indices visible to the labeler."""
    lines = []
    user_idx = 0
    for turn in conv.conversation:
        if turn.role == "user":
            lines.append(f"[user turn {user_idx}] USER: {turn.content}")
            user_idx += 1
        else:
            lines.append(f"ASSISTANT: {turn.content}")
    return "\n\n".join(lines)


def format_prompt(conv: Conversation) -> str:
    return PROMPT_TEMPLATE.replace(
        "{conversation_formatted}", format_conversation_for_labeler(conv)
    )


def load_existing_labeled_ids(path: Path) -> set:
    if not path.exists():
        return set()
    return {
        row["seed_id"]
        for row in load_jsonl(path)
        if row.get("turn_labels") is not None and "seed_id" in row
    }


async def label_one(
    conv: Conversation,
    cfg: LabelConfig,
    inference_api: InferenceAPI,
) -> Optional[Conversation]:
    user_prompt = format_prompt(conv)
    try:
        response: List[LLMResponse] = await inference_api(
            model_id=cfg.model_id,
            prompt=Prompt(messages=[ChatMessage(role=MessageRole.user, content=user_prompt)]),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        label_output: LabelingOutput = parse_and_validate_json(
            response[0].completion, LabelingOutput, allow_partial=True
        )

        n_user = sum(1 for t in conv.conversation if t.role == "user")
        if len(label_output.turns) != n_user:
            logger.warning(
                f"seed {conv.seed_id}: expected {n_user} labels, got {len(label_output.turns)}"
            )

        conv.turn_labels = label_output.turns
        save_jsonl(conv.model_dump(), cfg.output_path, append=True)
        return conv
    except Exception as e:
        logger.error(f"seed {conv.seed_id}: labeling failed: {e}")
        return None


async def main(cfg: LabelConfig):
    if not cfg.input_path.exists():
        raise FileNotFoundError(f"Input not found: {cfg.input_path}. Run generate.py first.")
    rows = load_jsonl(cfg.input_path)
    convs = [Conversation.model_validate(r) for r in rows]
    existing = load_existing_labeled_ids(cfg.output_path)
    todo = [c for c in convs if c.seed_id not in existing]
    logger.info(
        f"Total convs: {len(convs)} | already labeled: {len(existing)} | to label: {len(todo)}"
    )
    if not todo:
        return

    inference_api = InferenceAPI(
        cache_dir=cfg.safetytooling_cache_dir,
        anthropic_num_threads=cfg.max_concurrent_tasks,
    )

    tasks = [label_one(c, cfg, inference_api) for c in todo]
    results = await tqdm_asyncio.gather(*tasks, desc="Labeling")
    successful = sum(1 for r in results if r is not None)
    logger.info(f"Labeled {successful}/{len(todo)} conversations -> {cfg.output_path}")


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(LabelConfig, dest="config")
    args = parser.parse_args()
    asyncio.run(main(args.config))
