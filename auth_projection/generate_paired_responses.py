"""Generate the next assistant turn for each (parent, child) prefix from minedit pairs,
using identical sampling settings so the comparison is paired.

Run:
    uv run python -m auth_projection.generate_paired_responses
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

from utils.file_utils import load_jsonl
from utils.model_utils import get_device, load_model_and_tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_INPUT = Path(__file__).parent / "data" / "v1_minedit_pairs.jsonl"
DEFAULT_OUTPUT = Path(__file__).parent / "data" / "v1_minedit_assistant_responses.jsonl"


@torch.no_grad()
def generate_one(model, tokenizer, device, prefix: List[Dict], seed: int,
                 max_new_tokens: int = 250, temperature: float = 0.7) -> str:
    full = tokenizer.apply_chat_template(prefix, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(full, return_tensors="pt").to(device)
    torch.manual_seed(seed)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen = out[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=250)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    pairs = load_jsonl(args.input_path)
    if args.limit:
        pairs = pairs[: args.limit]
    logger.info(f"Loaded {len(pairs)} pairs")

    # Resume support
    done = set()
    if args.output_path.exists():
        for r in load_jsonl(args.output_path):
            done.add(r["pair_id"])
        logger.info(f"Resuming: {len(done)} pairs already generated")

    pairs = [p for p in pairs if p["pair_id"] not in done]
    if not pairs:
        logger.info("All pairs already generated")
        return

    logger.info(f"Loading {args.model_name}...")
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    model.eval()
    device = get_device()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(args.output_path, "a")
    try:
        for p in tqdm(pairs, desc="Generating pairs"):
            parent_resp = generate_one(
                model, tokenizer, device, p["parent_prefix"],
                seed=args.seed, max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            child_resp = generate_one(
                model, tokenizer, device, p["child_prefix"],
                seed=args.seed, max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            rec = {
                "pair_id": p["pair_id"],
                "parent_seed_id": p["parent_seed_id"],
                "child_seed_id": p["child_seed_id"],
                "user_turn_index": p["user_turn_index"],
                "flip_direction": p["flip_direction"],
                "parent_label": p["parent_label"],
                "child_label": p["child_label"],
                "topic": p["topic"],
                "parent_user_turn": p["parent_user_turn"],
                "child_user_turn": p["child_user_turn"],
                "parent_response": parent_resp,
                "child_response": child_resp,
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
    finally:
        f.close()
    logger.info(f"Saved -> {args.output_path}")


if __name__ == "__main__":
    main()
