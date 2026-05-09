"""A1 — verify the labeler is using preceding context (not just labeling each user turn in isolation).

For ~12 selected user turns, re-label twice:
  (a) with full preceding context — should match the original v1 label
  (b) with NO preceding context (just the user turn alone) — diagnoses how much
      the labeler's verdict depends on what came before

Key diagnostic: are tier verdicts different between (a) and (b)? If they're identical
for most turns, the labeler is mostly just reading the current turn's surface features
and our cumulative-labeling claim is weaker than it sounds. If they differ in expected
directions (e.g. polite-framing turns labeled `somewhat` in context but `none` in
isolation), cumulative labeling is doing real work.

Run:
    uv run python -m auth_projection.labeler_context_ablation
"""

import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import simple_parsing
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, LLMResponse, MessageRole, Prompt
from safetytooling.utils.experiment_utils import ExperimentConfigBase

from utils.file_utils import load_jsonl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
assert os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY is not set"


PROMPT_PATH = Path(__file__).parent / "prompts" / "labeler.prompt"
PROMPT_TEMPLATE = PROMPT_PATH.read_text().strip()


@dataclass
class Config(ExperimentConfigBase):
    model_id: str = "claude-sonnet-4-6"
    temperature: float = 0.0
    max_tokens: int = 1024
    max_concurrent_tasks: int = 5
    n_turns_to_test: int = 12
    seed: int = 11
    input_path: Path = Path(__file__).parent / "data" / "v1_labeled.jsonl"
    output_path: Path = Path(__file__).parent / "data" / "v1_labeler_context_ablation.json"
    output_dir: Path = Path(__file__).parent / "data"
    safetytooling_cache_dir: Union[str, Path] = Path.home() / ".safetytooling_cache"

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.input_path, str):
            self.input_path = Path(self.input_path)
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)
        if isinstance(self.safetytooling_cache_dir, str):
            self.safetytooling_cache_dir = Path(self.safetytooling_cache_dir)


def select_turns(convs: List[Dict], n: int, seed: int) -> List[Tuple[Dict, int]]:
    """Pick a diverse set of (conversation, user_turn_index) pairs.
    Prefer non-first turns (so preceding-context matters) and mix tiers."""
    rng = random.Random(seed)
    candidates_by_label = {"none": [], "somewhat": [], "strongly": []}
    for conv in convs:
        for tl in conv.get("turn_labels") or []:
            t = tl["turn_index"]
            if t == 0:
                continue  # first turn has no preceding context to ablate
            candidates_by_label[tl["label"]].append((conv, t, tl["label"]))
    # Round-robin sample by label so we get diversity
    selected = []
    rng.shuffle(candidates_by_label["none"])
    rng.shuffle(candidates_by_label["somewhat"])
    rng.shuffle(candidates_by_label["strongly"])
    while len(selected) < n:
        for label in ["none", "somewhat", "strongly"]:
            if candidates_by_label[label]:
                selected.append(candidates_by_label[label].pop())
                if len(selected) >= n:
                    break
    return selected


def render_with_context(conv: Dict, target_turn: int) -> str:
    """Render conversation up through target_turn (inclusive), as the labeler sees it."""
    out = []
    user_idx = 0
    for turn in conv["conversation"]:
        if turn["role"] == "user":
            out.append(f"[user turn {user_idx}] USER: {turn['content']}")
            if user_idx == target_turn:
                break
            user_idx += 1
        else:
            out.append(f"ASSISTANT: {turn['content']}")
    return "\n\n".join(out)


def render_no_context(conv: Dict, target_turn: int) -> str:
    """Render only the target user turn, with index 0, as if it were a single-turn conv."""
    user_idx = 0
    for turn in conv["conversation"]:
        if turn["role"] == "user":
            if user_idx == target_turn:
                return f"[user turn 0] USER: {turn['content']}"
            user_idx += 1
    raise ValueError(f"target_turn={target_turn} not found in conv")


async def label_one(
    formatted_conv: str,
    cfg: Config,
    inference_api: InferenceAPI,
) -> Dict:
    user_prompt = PROMPT_TEMPLATE.replace("{conversation_formatted}", formatted_conv)
    response: List[LLMResponse] = await inference_api(
        model_id=cfg.model_id,
        prompt=Prompt(messages=[ChatMessage(role=MessageRole.user, content=user_prompt)]),
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
    raw = response[0].completion.strip()
    # Strip code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:].strip()
    return json.loads(raw)


async def main(cfg: Config):
    convs = load_jsonl(cfg.input_path)
    selected = select_turns(convs, cfg.n_turns_to_test, cfg.seed)
    logger.info(f"Selected {len(selected)} turns to ablate")

    inference_api = InferenceAPI(
        cache_dir=cfg.safetytooling_cache_dir,
        anthropic_num_threads=cfg.max_concurrent_tasks,
    )

    async def relabel_pair(conv, target_turn, original_label):
        ctx_formatted = render_with_context(conv, target_turn)
        nctx_formatted = render_no_context(conv, target_turn)
        try:
            with_ctx = await label_one(ctx_formatted, cfg, inference_api)
            no_ctx = await label_one(nctx_formatted, cfg, inference_api)
        except Exception as e:
            logger.error(f"label failed for seed={conv['seed_id'][:8]} t={target_turn}: {e}")
            return None
        # The with-context labeler returns labels for ALL turns up through target;
        # we want the last one. The no-context labeler should return one entry.
        with_ctx_label = with_ctx["turns"][-1]["label"] if with_ctx.get("turns") else None
        no_ctx_label = no_ctx["turns"][0]["label"] if no_ctx.get("turns") else None
        return {
            "seed_id": conv["seed_id"],
            "target_turn": target_turn,
            "user_turn_text": next(
                t["content"] for i, t in enumerate(conv["conversation"])
                if t["role"] == "user" and sum(
                    1 for tt in conv["conversation"][:i] if tt["role"] == "user"
                ) == target_turn
            )[:200],
            "original_label": original_label,
            "with_context_label": with_ctx_label,
            "no_context_label": no_ctx_label,
            "with_context_rationale": with_ctx["turns"][-1].get("rationale", "") if with_ctx.get("turns") else "",
            "no_context_rationale": no_ctx["turns"][0].get("rationale", "") if no_ctx.get("turns") else "",
            "lexical_twin_kind": conv.get("lexical_twin_kind"),
        }

    coros = [relabel_pair(c, t, lbl) for (c, t, lbl) in selected]
    results = await tqdm_asyncio.gather(*coros, desc="Relabeling")
    results = [r for r in results if r is not None]

    # Aggregate
    n = len(results)
    n_orig_match = sum(1 for r in results if r["with_context_label"] == r["original_label"])
    n_ctx_vs_nctx_match = sum(1 for r in results if r["with_context_label"] == r["no_context_label"])
    n_differ = n - n_ctx_vs_nctx_match

    summary = {
        "n_turns_tested": n,
        "with_context_matches_original": n_orig_match,
        "context_vs_no_context_match": n_ctx_vs_nctx_match,
        "context_changes_label": n_differ,
        "rate_context_changes_label": n_differ / n if n else 0.0,
        "results": results,
    }

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Cumulative-labeling spot-check (n={n}) ===")
    print(f"with-context label matches original v1 label: {n_orig_match}/{n} ({100*n_orig_match/n:.0f}%) "
          f"— sanity check that re-labeling is reproducible")
    print(f"context vs no-context label match: {n_ctx_vs_nctx_match}/{n} ({100*n_ctx_vs_nctx_match/n:.0f}%)")
    print(f"context CHANGES the label on: {n_differ}/{n} turns ({100*n_differ/n:.0f}%)")
    print(f"  → if this is high (>30%), labeler is genuinely using context.")
    print(f"  → if low (<10%), labeler is mostly reading the current turn in isolation.")
    print()
    print("Per-turn detail:")
    for r in results:
        marker = " ← CHANGED" if r["with_context_label"] != r["no_context_label"] else ""
        print(f"  seed={r['seed_id'][:8]} turn={r['target_turn']} "
              f"orig={r['original_label']:<9} ctx={r['with_context_label']:<9} "
              f"no_ctx={r['no_context_label']:<9}{marker}")
        print(f"    text: {r['user_turn_text'][:140]}")
    logger.info(f"Saved -> {cfg.output_path}")


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(Config, dest="config")
    args = parser.parse_args()
    asyncio.run(main(args.config))
