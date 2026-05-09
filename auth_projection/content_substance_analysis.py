"""For each held-out conversation in the B4 steering output, ask Sonnet whether
the assistant's actual ADVICE / RECOMMENDATION shifts substantively across
steering alphas (or whether it's just stylistic — same advice, different framing).

This goes beyond length/hedging/directiveness style metrics. The question is:
  Does steering change the ACTION the model recommends, or just the wrapper?

For each conversation we send Sonnet the conversation prefix + all coherent-α
responses. Sonnet outputs:
  - per-α verdict vs the α=0 baseline:
      SAME_ADVICE / DIFFERENT_FRAMING_SAME_ACTION / DIFFERENT_RECOMMENDATION
  - 1-sentence summary of how the advice shifted (if at all)

Run:
    uv run python -m auth_projection.content_substance_analysis \\
        --steering_path auth_projection/data/v4a_steering_real_conversations.json \\
        --output_path auth_projection/data/v4a_content_substance.jsonl \\
        --coherent_alphas=-2,-1,0,1,2
"""

import argparse
import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import simple_parsing
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm_asyncio
from typing import Literal

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, LLMResponse, MessageRole, Prompt
from safetytooling.utils.experiment_utils import ExperimentConfigBase

from utils.file_utils import load_jsonl, save_jsonl
from utils.parsing import parse_and_validate_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
assert os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY is not set"


PROMPT = """You are evaluating whether two assistant responses recommend SUBSTANTIVELY DIFFERENT actions, or whether they recommend the same action with different framing.

CONVERSATION CONTEXT:
{prefix}

RESPONSE A (baseline):
{response_a}

RESPONSE B (test):
{response_b}

Read both carefully. The question is NOT about style (length, tone, hedging) — it's about the underlying recommendation. Choose ONE verdict:

- **SAME_ADVICE**: both responses recommend essentially the same course of action. Wording differs but the user would do the same thing if they followed either.

- **DIFFERENT_FRAMING_SAME_ACTION**: both responses point toward the same concrete action(s) but frame them differently — e.g., one is more directive, one is more enumerative; one is more confident, one is more hedged. Same destination, different packaging.

- **DIFFERENT_RECOMMENDATION**: the responses recommend substantively different actions/decisions. Following A vs B would lead the user to do different things in the world.

Output JSON only:

{{
  "verdict": "SAME_ADVICE" | "DIFFERENT_FRAMING_SAME_ACTION" | "DIFFERENT_RECOMMENDATION",
  "summary": "<one sentence: what is the substantive content difference, if any?>",
  "confidence": <float 0.0-1.0>
}}
"""


class ContentVerdict(BaseModel):
    verdict: Literal["SAME_ADVICE", "DIFFERENT_FRAMING_SAME_ACTION", "DIFFERENT_RECOMMENDATION"]
    summary: str
    confidence: float = Field(ge=0.0, le=1.0)


@dataclass
class Config(ExperimentConfigBase):
    model_id: str = "claude-sonnet-4-6"
    temperature: float = 0.0
    max_tokens: int = 600
    max_concurrent_tasks: int = 10
    steering_path: Path = Path("auth_projection/data/v4a_steering_real_conversations.json")
    convs_path: Path = Path("auth_projection/data/v1_labeled.jsonl")
    output_path: Path = Path("auth_projection/data/v4a_content_substance.jsonl")
    output_dir: Path = Path("auth_projection/data")
    coherent_alphas: str = "-2,-1,0,1,2"  # comma-sep; 0 must be in this set (it's the baseline)
    safetytooling_cache_dir: Union[str, Path] = Path.home() / ".safetytooling_cache"

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.steering_path, str):
            self.steering_path = Path(self.steering_path)
        if isinstance(self.convs_path, str):
            self.convs_path = Path(self.convs_path)
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)
        if isinstance(self.safetytooling_cache_dir, str):
            self.safetytooling_cache_dir = Path(self.safetytooling_cache_dir)


def render_prefix(seed_id: str, last_user_text: str, convs_by_seed: Dict) -> str:
    conv = convs_by_seed.get(seed_id)
    if conv is None:
        return f"[USER]: {last_user_text}"
    out = []
    for t in conv["conversation"]:
        out.append(f"[{t['role'].upper()}]: {t['content']}")
        if t["role"] == "user" and t["content"] == last_user_text:
            break
    return "\n\n".join(out)


async def compare_one(prefix: str, response_a: str, response_b: str,
                      cfg: Config, inference_api: InferenceAPI) -> Optional[Dict]:
    prompt_text = PROMPT.format(prefix=prefix, response_a=response_a, response_b=response_b)
    try:
        response: List[LLMResponse] = await inference_api(
            model_id=cfg.model_id,
            prompt=Prompt(messages=[ChatMessage(role=MessageRole.user, content=prompt_text)]),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        out = parse_and_validate_json(response[0].completion, ContentVerdict, allow_partial=True)
        return {"verdict": out.verdict, "summary": out.summary, "confidence": out.confidence}
    except Exception as e:
        logger.error(f"Compare failed: {e}")
        return None


async def main(cfg: Config):
    samples = json.load(open(cfg.steering_path))
    convs = load_jsonl(cfg.convs_path)
    convs_by_seed = {c["seed_id"]: c for c in convs}

    coherent = [float(x) for x in cfg.coherent_alphas.split(",") if x.strip()]
    assert 0.0 in coherent, "α=0 (baseline) must be in coherent_alphas"
    nonzero_alphas = [a for a in coherent if a != 0.0]

    # Index samples by (seed_id, alpha)
    by_key = {}
    for s in samples:
        by_key[(s["seed_id"], s["alpha"])] = s

    # Build comparison tasks: for each conv, compare α=0 to each non-zero α
    seed_ids = sorted({s["seed_id"] for s in samples})
    logger.info(f"Conversations: {len(seed_ids)}")
    logger.info(f"Coherent alphas: {coherent} (compare each non-zero to baseline α=0)")

    # Resume support
    done = set()
    if cfg.output_path.exists():
        for r in load_jsonl(cfg.output_path):
            done.add((r["seed_id"], r["alpha"]))
        logger.info(f"Resuming: {len(done)} already analyzed")

    inference_api = InferenceAPI(
        cache_dir=cfg.safetytooling_cache_dir,
        anthropic_num_threads=cfg.max_concurrent_tasks,
    )

    tasks = []
    metadata = []
    for seed in seed_ids:
        baseline = by_key.get((seed, 0.0))
        if baseline is None:
            continue
        prefix = render_prefix(seed, baseline["last_user_text"], convs_by_seed)
        for alpha in nonzero_alphas:
            if (seed, alpha) in done:
                continue
            test = by_key.get((seed, alpha))
            if test is None:
                continue
            tasks.append(compare_one(prefix, baseline["completion"], test["completion"], cfg, inference_api))
            metadata.append({
                "seed_id": seed,
                "alpha": alpha,
                "topic": baseline.get("topic"),
                "last_user_label": baseline.get("last_user_label"),
                "last_user_text_preview": (baseline.get("last_user_text") or "")[:160],
            })

    logger.info(f"Submitting {len(tasks)} pairwise comparisons")
    if not tasks:
        logger.info("Nothing to do")
        return

    results = await tqdm_asyncio.gather(*tasks, desc="Comparing")
    n_ok = 0
    for meta, res in zip(metadata, results):
        if res is None:
            continue
        save_jsonl({**meta, **res}, cfg.output_path, append=True)
        n_ok += 1
    logger.info(f"Saved {n_ok}/{len(tasks)} -> {cfg.output_path}")

    # Quick summary
    print("\n=== Verdict distribution ===")
    by_alpha = {}
    for r in load_jsonl(cfg.output_path):
        a = r["alpha"]; v = r["verdict"]
        by_alpha.setdefault(a, {}).setdefault(v, 0)
        by_alpha[a][v] += 1
    print(f"{'α':<6} {'SAME_ADVICE':<14} {'DIFF_FRAMING':<14} {'DIFF_RECOMMEND':<16} {'total':<6}")
    for a in sorted(by_alpha.keys()):
        c = by_alpha[a]
        total = sum(c.values())
        print(f"{a:+5.1f}  {c.get('SAME_ADVICE',0):<14} {c.get('DIFFERENT_FRAMING_SAME_ACTION',0):<14} "
              f"{c.get('DIFFERENT_RECOMMENDATION',0):<16} {total:<6}")


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(Config, dest="config")
    args = parser.parse_args()
    asyncio.run(main(args.config))
