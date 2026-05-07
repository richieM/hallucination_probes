"""Aggregate judge results into per-pair verdict and per-flip-direction win rates."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from utils.file_utils import load_jsonl


JUDGE_PATH = Path(__file__).parent / "data" / "v1_minedit_judge.jsonl"
OUT_PATH = Path(__file__).parent / "data" / "v1_minedit_judge_summary.json"


def signed_flip(direction: str) -> int:
    rank = {"none": 0, "somewhat": 1, "strongly": 2}
    parent, child = direction.split("->")
    return int(np.sign(rank[child] - rank[parent]))


def main():
    rows = load_jsonl(JUDGE_PATH)
    by_pair = defaultdict(list)
    for r in rows:
        by_pair[r["pair_id"]].append(r)

    n_pairs = len(by_pair)
    print(f"Pairs with judgments: {n_pairs}")

    pair_verdicts = []  # (pair_id, flip_direction, signed_flip, position_bias_consistent, agreed_winner, n_judgments)
    position_bias_count = Counter()
    consistency_count = Counter()
    for pair_id, judgments in by_pair.items():
        flip = judgments[0]["flip_direction"]
        sf = signed_flip(flip)
        winners_resolved = [j["winner_resolved"] for j in judgments]
        # position-by-A check: does swapping A/B change the verdict?
        wins_when_a_is_parent = [j for j in judgments if j["a_is_parent"]]
        wins_when_a_is_child = [j for j in judgments if not j["a_is_parent"]]

        # Map: same winner regardless of position?
        if wins_when_a_is_parent and wins_when_a_is_child:
            same = wins_when_a_is_parent[0]["winner_resolved"] == wins_when_a_is_child[0]["winner_resolved"]
            consistency_count["consistent" if same else "inconsistent"] += 1
            # Count "raw_winner=A" tendency (position bias)
            for j in judgments:
                position_bias_count[j["raw_winner"]] += 1

        # Final verdict: take majority across position swaps
        c = Counter(winners_resolved)
        if c.most_common(1)[0][1] == len(judgments):
            verdict = c.most_common(1)[0][0]
        else:
            # Disagreement -> tie
            verdict = "tie"
        pair_verdicts.append({
            "pair_id": pair_id,
            "flip_direction": flip,
            "signed_flip": sf,
            "verdict": verdict,
            "raw_winners": winners_resolved,
            "consistent": same if (wins_when_a_is_parent and wins_when_a_is_child) else None,
        })

    print(f"\nPosition-swap consistency:")
    for k, v in consistency_count.items():
        print(f"  {k}: {v}")
    print(f"Raw position bias (raw_winner across all calls):")
    for k, v in position_bias_count.most_common():
        print(f"  {k}: {v}")

    # Aggregate by signed flip direction
    print(f"\n{'flip_signed':<12} {'n':<4} {'parent_wins':<12} {'child_wins':<12} {'tie':<6}")
    by_sf = defaultdict(list)
    for v in pair_verdicts:
        by_sf[v["signed_flip"]].append(v["verdict"])
    sf_summary = {}
    for sf in sorted(by_sf):
        verdicts = by_sf[sf]
        n = len(verdicts)
        c = Counter(verdicts)
        sf_summary[str(sf)] = {
            "n": n,
            "parent_wins": c.get("parent", 0),
            "child_wins": c.get("child", 0),
            "ties": c.get("tie", 0),
            "parent_win_rate": c.get("parent", 0) / n,
            "child_win_rate": c.get("child", 0) / n,
        }
        print(f"{sf:<12} {n:<4} {c.get('parent', 0):<12} {c.get('child', 0):<12} {c.get('tie', 0):<6}")

    # Per-direction breakdown
    by_dir = defaultdict(list)
    for v in pair_verdicts:
        by_dir[v["flip_direction"]].append(v["verdict"])
    print(f"\nPer flip direction:")
    print(f"{'direction':<24} {'n':<4} {'parent':<8} {'child':<8} {'tie':<5}")
    dir_summary = {}
    for direction in sorted(by_dir):
        verdicts = by_dir[direction]
        n = len(verdicts)
        c = Counter(verdicts)
        dir_summary[direction] = {
            "n": n,
            "parent_wins": c.get("parent", 0),
            "child_wins": c.get("child", 0),
            "ties": c.get("tie", 0),
        }
        print(f"{direction:<24} {n:<4} {c.get('parent', 0):<8} {c.get('child', 0):<8} {c.get('tie', 0):<5}")

    summary = {
        "n_pairs": n_pairs,
        "consistency": dict(consistency_count),
        "position_bias_raw_winner": dict(position_bias_count),
        "by_signed_flip": sf_summary,
        "by_direction": dir_summary,
        "pair_verdicts": pair_verdicts,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
