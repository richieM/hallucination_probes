"""Blind LLM-judge comparison of (parent, child) assistant response pairs.

For each pair we run the judge twice with positions swapped (parent->A, child->A) so
we can measure position bias and average it out. The judge sees the conversation
prefixes (so it can ground the judgment) but is asked to evaluate directiveness/
hedging/stance, not to identify which prompt is which.

Run:
    uv run python -m auth_projection.judge_pairs
"""

import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import simple_parsing
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, LLMResponse, MessageRole, Prompt
from safetytooling.utils.experiment_utils import ExperimentConfigBase

from utils.file_utils import load_jsonl, save_jsonl
from utils.parsing import parse_and_validate_json

from pydantic import BaseModel, Field
from typing import Literal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
assert os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY is not set"


PROMPT_PATH = Path(__file__).parent / "prompts" / "judge_directiveness.prompt"
PROMPT_TEMPLATE = PROMPT_PATH.read_text().strip()


class JudgeOutput(BaseModel):
    winner: Literal["A", "B", "tie"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


@dataclass
class JudgeConfig(ExperimentConfigBase):
    model_id: str = "claude-sonnet-4-6"
    temperature: float = 0.0
    max_tokens: int = 512
    max_concurrent_tasks: int = 10

    input_path: Path = Path(__file__).parent / "data" / "v1_minedit_assistant_responses.jsonl"
    output_path: Path = Path(__file__).parent / "data" / "v1_minedit_judge.jsonl"
    output_dir: Path = Path(__file__).parent / "data"

    datetime_str: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    safetytooling_cache_dir: Union[str, Path] = Path.home() / ".safetytooling_cache"
    save_path: Optional[Path] = None
    log_to_file: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.input_path = Path(self.input_path)
        self.output_path = Path(self.output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(self.safetytooling_cache_dir, str):
            self.safetytooling_cache_dir = Path(self.safetytooling_cache_dir)


def render_prefix(prefix: List[Dict]) -> str:
    out = []
    for t in prefix:
        role = t["role"].upper()
        out.append(f"{role}: {t['content']}")
    return "\n\n".join(out)


def build_prompt(prefix_x: List[Dict], response_a: str,
                 prefix_y: List[Dict], response_b: str) -> str:
    return (
        PROMPT_TEMPLATE
        .replace("{prefix_x}", render_prefix(prefix_x))
        .replace("{prefix_y}", render_prefix(prefix_y))
        .replace("{response_a}", response_a)
        .replace("{response_b}", response_b)
    )


async def judge_one(
    pair: Dict,
    a_is_parent: bool,
    cfg: JudgeConfig,
    inference_api: InferenceAPI,
) -> Optional[Dict]:
    """If a_is_parent: A=parent, B=child. Else: A=child, B=parent."""
    if a_is_parent:
        prefix_a = pair["parent_prefix"]; resp_a = pair["parent_response"]
        prefix_b = pair["child_prefix"];  resp_b = pair["child_response"]
    else:
        prefix_a = pair["child_prefix"];  resp_a = pair["child_response"]
        prefix_b = pair["parent_prefix"]; resp_b = pair["parent_response"]

    prompt_text = build_prompt(prefix_a, resp_a, prefix_b, resp_b)
    try:
        response: List[LLMResponse] = await inference_api(
            model_id=cfg.model_id,
            prompt=Prompt(messages=[ChatMessage(role=MessageRole.user, content=prompt_text)]),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        out: JudgeOutput = parse_and_validate_json(response[0].completion, JudgeOutput, allow_partial=True)
        winner_actual = ("parent" if out.winner == "A" else "child" if out.winner == "B" else "tie") \
            if a_is_parent else \
            ("child" if out.winner == "A" else "parent" if out.winner == "B" else "tie")
        return {
            "pair_id": pair["pair_id"],
            "a_is_parent": a_is_parent,
            "raw_winner": out.winner,
            "winner_resolved": winner_actual,
            "confidence": out.confidence,
            "rationale": out.rationale,
            "flip_direction": pair["flip_direction"],
            "parent_label": pair["parent_label"],
            "child_label": pair["child_label"],
        }
    except Exception as e:
        logger.error(f"pair {pair['pair_id']} (a_is_parent={a_is_parent}): {e}")
        return None


async def main(cfg: JudgeConfig):
    pairs_with_responses = load_jsonl(cfg.input_path)
    # need to also load the original prefixes (not stored in responses file)
    pairs_path = Path(__file__).parent / "data" / "v1_minedit_pairs.jsonl"
    prefix_by_id = {p["pair_id"]: p for p in load_jsonl(pairs_path)}
    enriched = []
    for r in pairs_with_responses:
        p = prefix_by_id.get(r["pair_id"])
        if p is None:
            continue
        enriched.append({
            **r,
            "parent_prefix": p["parent_prefix"],
            "child_prefix": p["child_prefix"],
        })
    logger.info(f"Loaded {len(enriched)} response pairs (with prefixes)")

    # Resume
    done = set()
    if cfg.output_path.exists():
        for r in load_jsonl(cfg.output_path):
            done.add((r["pair_id"], r["a_is_parent"]))
        logger.info(f"Resuming: {len(done)} judge calls already done")

    tasks_to_run = []
    for pair in enriched:
        for a_is_parent in [True, False]:
            if (pair["pair_id"], a_is_parent) in done:
                continue
            tasks_to_run.append((pair, a_is_parent))
    logger.info(f"Submitting {len(tasks_to_run)} judge calls")

    if not tasks_to_run:
        return

    inference_api = InferenceAPI(
        cache_dir=cfg.safetytooling_cache_dir,
        anthropic_num_threads=cfg.max_concurrent_tasks,
    )

    coros = [judge_one(pair, aip, cfg, inference_api) for (pair, aip) in tasks_to_run]
    results = await tqdm_asyncio.gather(*coros, desc="Judging")

    n_ok = 0
    for res in results:
        if res is None:
            continue
        save_jsonl(res, cfg.output_path, append=True)
        n_ok += 1
    logger.info(f"Saved {n_ok}/{len(tasks_to_run)} judgments -> {cfg.output_path}")


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(JudgeConfig, dest="config")
    args = parser.parse_args()
    asyncio.run(main(args.config))
