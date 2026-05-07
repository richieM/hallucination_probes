"""Build (parent, child) prefix pairs at every turn where the labeler's label flipped.

For each minimal-edit child whose parent is in the v1 labeled set, walk per-user-turn
labels in lockstep. Wherever labels differ at the same user_turn_index t, record:

  - parent_prefix: conversation through user turn t (i.e. user turns 0..t and assistant
    turns 0..t-1)
  - child_prefix: same shape from the child conv
  - parent_label, child_label, flip_direction (e.g. "none->strongly")

These prefixes feed Step 2 (paired assistant generation) — the model generates the next
assistant reply for both versions, and we compare.

Run:
    uv run python -m auth_projection.build_minedit_pairs
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

from utils.file_utils import load_jsonl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_PARENT = Path(__file__).parent / "data" / "v1_labeled.jsonl"
DEFAULT_CHILD = Path(__file__).parent / "data" / "v1_cf_minimal_edit_labeled.jsonl"
DEFAULT_OUTPUT = Path(__file__).parent / "data" / "v1_minedit_pairs.jsonl"


def conv_prefix_through_user_turn(conversation: List[Dict], user_turn_index: int) -> List[Dict]:
    """Return list of {role, content} dicts up to and including the (user_turn_index)-th
    user turn. Assumes alternating roles starting with user."""
    out = []
    user_count = 0
    for t in conversation:
        out.append({"role": t["role"], "content": t["content"]})
        if t["role"] == "user":
            if user_count == user_turn_index:
                return out
            user_count += 1
    raise ValueError(
        f"user_turn_index={user_turn_index} not found; only {user_count} user turns"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_path", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--child_path", type=Path, default=DEFAULT_CHILD)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    parents = {r["seed_id"]: r for r in load_jsonl(args.parent_path)}
    children = load_jsonl(args.child_path)
    logger.info(f"Loaded {len(parents)} parents, {len(children)} children")

    pairs = []
    pair_id = 0
    for child in children:
        parent_seed = child.get("parent_seed_id") or child["seed_id"].split("__minedit_to_")[0]
        parent = parents.get(parent_seed)
        if parent is None:
            logger.warning(f"No parent for child {child['seed_id']}")
            continue

        parent_labels = {tl["turn_index"]: tl["label"] for tl in (parent.get("turn_labels") or [])}
        child_labels = {tl["turn_index"]: tl["label"] for tl in (child.get("turn_labels") or [])}

        common_turns = sorted(set(parent_labels) & set(child_labels))
        for t in common_turns:
            if parent_labels[t] == child_labels[t]:
                continue
            try:
                p_prefix = conv_prefix_through_user_turn(parent["conversation"], t)
                c_prefix = conv_prefix_through_user_turn(child["conversation"], t)
            except ValueError as e:
                logger.warning(f"Skip {child['seed_id']} t={t}: {e}")
                continue

            pairs.append({
                "pair_id": pair_id,
                "parent_seed_id": parent_seed,
                "child_seed_id": child["seed_id"],
                "user_turn_index": t,
                "parent_label": parent_labels[t],
                "child_label": child_labels[t],
                "flip_direction": f"{parent_labels[t]}->{child_labels[t]}",
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
    logger.info(f"Wrote {len(pairs)} pair records -> {args.output_path}")

    from collections import Counter
    flip_counts = Counter(p["flip_direction"] for p in pairs)
    logger.info(f"Flip directions: {dict(flip_counts)}")


if __name__ == "__main__":
    main()
