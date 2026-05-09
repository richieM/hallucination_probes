"""Build (parent, paraphrase-child) prefix pairs at user turns where the labeler kept
the same tier label across the paraphrase. The same-tier baseline for the substance
analysis: parent and paraphrase have similar magnitude of text difference as
minimal-edit pairs but DON'T flip user-state.

Output JSONL has the same schema as build_minedit_pairs so the downstream
generate_paired_responses + content_substance_minedit scripts work unchanged.

Run:
    uv run python -m auth_projection.build_paraphrase_pairs
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

from utils.file_utils import load_jsonl


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def conv_prefix_through_user_turn(conversation: List[Dict], user_turn_index: int) -> List[Dict]:
    out = []
    user_count = 0
    for t in conversation:
        out.append({"role": t["role"], "content": t["content"]})
        if t["role"] == "user":
            if user_count == user_turn_index:
                return out
            user_count += 1
    raise ValueError(f"user_turn_index={user_turn_index} not found")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_path", type=Path,
                        default=Path("auth_projection/data/v1_labeled.jsonl"))
    parser.add_argument("--child_path", type=Path,
                        default=Path("auth_projection/data/v1_cf_paraphrase_labeled.jsonl"))
    parser.add_argument("--output_path", type=Path,
                        default=Path("auth_projection/data/v1_paraphrase_pairs.jsonl"))
    args = parser.parse_args()

    parents = {r["seed_id"]: r for r in load_jsonl(args.parent_path)}
    children = load_jsonl(args.child_path)
    logger.info(f"Loaded {len(parents)} parents, {len(children)} paraphrase children")

    pairs = []
    pair_id = 0
    for child in children:
        parent_seed = child.get("parent_seed_id") or child["seed_id"].split("__para")[0]
        parent = parents.get(parent_seed)
        if parent is None:
            continue
        parent_labels = {tl["turn_index"]: tl["label"] for tl in (parent.get("turn_labels") or [])}
        child_labels = {tl["turn_index"]: tl["label"] for tl in (child.get("turn_labels") or [])}
        # Keep ONLY user turns where labels match (same-tier baseline)
        common = sorted(set(parent_labels) & set(child_labels))
        for t in common:
            if parent_labels[t] != child_labels[t]:
                continue  # flipped — skip
            try:
                p_prefix = conv_prefix_through_user_turn(parent["conversation"], t)
                c_prefix = conv_prefix_through_user_turn(child["conversation"], t)
            except ValueError:
                continue
            label = parent_labels[t]
            pairs.append({
                "pair_id": pair_id,
                "parent_seed_id": parent_seed,
                "child_seed_id": child["seed_id"],
                "user_turn_index": t,
                "parent_label": label,
                "child_label": label,
                "flip_direction": f"{label}->{label}",  # no flip
                "topic": parent.get("topic"),
                "parent_prefix": p_prefix,
                "child_prefix": c_prefix,
                "parent_user_turn": p_prefix[-1]["content"],
                "child_user_turn": c_prefix[-1]["content"],
            })
            pair_id += 1

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    logger.info(f"Wrote {len(pairs)} same-tier paraphrase pair records -> {args.output_path}")
    from collections import Counter
    print(f"Tier distribution: {Counter(p['parent_label'] for p in pairs)}")


if __name__ == "__main__":
    main()
