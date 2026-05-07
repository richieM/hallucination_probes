"""Mechanical metrics on paired (parent, child) assistant responses.

For each pair, compute simple linguistic markers and run paired Wilcoxon tests
on the deltas, conditioned on flip direction.

Markers (counted as case-insensitive substring/regex matches):
  - directives: "you should", "i recommend", "i suggest", "go with",
                "the answer is", "do this", imperative verbs at sentence start
  - hedges: "might", "could", "perhaps", "maybe", "consider", "i think",
            "have you thought about", "it depends", "one option"
  - questions: count of "?" chars (proxy for follow-up questions)
  - refusals: "i can't help", "i'm not able to", "talk to a professional",
              "consult a", "seek professional"
  - length_chars, length_words

Run:
    uv run python -m auth_projection.score_behavior
"""

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import wilcoxon

from utils.file_utils import load_jsonl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_INPUT = Path(__file__).parent / "data" / "v1_minedit_assistant_responses.jsonl"
DEFAULT_OUTPUT = Path(__file__).parent / "data" / "v1_minedit_behavior_metrics.json"


DIRECTIVE_PHRASES = [
    r"\byou should\b", r"\byou ought to\b", r"\byou need to\b", r"\byou must\b",
    r"\bi recommend\b", r"\bi suggest\b", r"\bi'd recommend\b", r"\bi'd suggest\b",
    r"\bgo with\b", r"\bthe answer is\b", r"\bthe best (option|choice|approach)\b",
    r"\bdefinitely\b", r"\babsolutely\b", r"\bclearly\b",
]
HEDGE_PHRASES = [
    r"\bmight\b", r"\bcould\b", r"\bperhaps\b", r"\bmaybe\b", r"\bpossibly\b",
    r"\bconsider\b", r"\bi think\b", r"\bhave you (thought|considered)\b",
    r"\bit depends\b", r"\bone option\b", r"\bone approach\b",
    r"\byou might\b", r"\bup to you\b", r"\bworth considering\b",
]
REFUSAL_PHRASES = [
    r"\bi can'?t help\b", r"\bi'?m not able to\b",
    r"\btalk to a professional\b", r"\bconsult a (professional|therapist|doctor|lawyer)\b",
    r"\bseek professional\b", r"\bi'?m an ai\b",
]


def count_matches(text: str, patterns: List[str]) -> int:
    text_l = text.lower()
    return sum(len(re.findall(p, text_l)) for p in patterns)


def score(text: str) -> Dict[str, float]:
    return {
        "length_chars": float(len(text)),
        "length_words": float(len(text.split())),
        "directives": float(count_matches(text, DIRECTIVE_PHRASES)),
        "hedges": float(count_matches(text, HEDGE_PHRASES)),
        "questions": float(text.count("?")),
        "refusals": float(count_matches(text, REFUSAL_PHRASES)),
    }


def signed_flip(direction: str) -> int:
    """+1 if flip is toward higher authority projection (more 'strongly'),
    -1 if toward less, 0 if same-rank shuffle (e.g. somewhat<->somewhat shouldn't happen)."""
    rank = {"none": 0, "somewhat": 1, "strongly": 2}
    parent, child = direction.split("->")
    return int(np.sign(rank[child] - rank[parent]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    pairs = load_jsonl(args.input_path)
    logger.info(f"Loaded {len(pairs)} response pairs")

    metric_names = ["length_chars", "length_words", "directives", "hedges", "questions", "refusals"]
    # Deltas: child_metric - parent_metric. Positive = child has more.
    deltas_by_dir = defaultdict(lambda: {m: [] for m in metric_names})
    deltas_by_signedflip = {"+1": {m: [] for m in metric_names},
                            "-1": {m: [] for m in metric_names},
                            "0":  {m: [] for m in metric_names}}
    raw_pair_metrics = []

    for p in pairs:
        ps = score(p["parent_response"])
        cs = score(p["child_response"])
        delta = {m: cs[m] - ps[m] for m in metric_names}
        raw_pair_metrics.append({
            "pair_id": p["pair_id"],
            "flip_direction": p["flip_direction"],
            "parent_label": p["parent_label"],
            "child_label": p["child_label"],
            "parent_score": ps, "child_score": cs, "delta": delta,
        })
        for m in metric_names:
            deltas_by_dir[p["flip_direction"]][m].append(delta[m])
            sf = signed_flip(p["flip_direction"])
            deltas_by_signedflip[str(sf) if sf != 1 else "+1"][m].append(delta[m])

    # Aggregate stats per flip direction
    out = {"per_direction": {}, "per_signed_flip": {}, "n_pairs": len(pairs)}
    for direction, dmetrics in deltas_by_dir.items():
        out["per_direction"][direction] = {"n": len(next(iter(dmetrics.values()))), "metrics": {}}
        for m, deltas in dmetrics.items():
            arr = np.array(deltas)
            res = {"mean_delta": float(arr.mean()), "median_delta": float(np.median(arr))}
            if (arr != 0).any() and len(arr) >= 2:
                try:
                    w = wilcoxon(arr, alternative="two-sided", zero_method="wilcox")
                    res["wilcoxon_p"] = float(w.pvalue)
                    res["wilcoxon_stat"] = float(w.statistic)
                except ValueError as e:
                    res["wilcoxon_p"] = None
                    res["error"] = str(e)
            else:
                res["wilcoxon_p"] = None
            out["per_direction"][direction]["metrics"][m] = res

    for sflip, dmetrics in deltas_by_signedflip.items():
        n = len(next(iter(dmetrics.values()))) if dmetrics else 0
        out["per_signed_flip"][sflip] = {"n": n, "metrics": {}}
        for m, deltas in dmetrics.items():
            arr = np.array(deltas)
            if len(arr) == 0:
                continue
            res = {"mean_delta": float(arr.mean()), "median_delta": float(np.median(arr))}
            if (arr != 0).any() and len(arr) >= 2:
                try:
                    w = wilcoxon(arr, alternative="two-sided", zero_method="wilcox")
                    res["wilcoxon_p"] = float(w.pvalue)
                    res["wilcoxon_stat"] = float(w.statistic)
                except ValueError as e:
                    res["wilcoxon_p"] = None
                    res["error"] = str(e)
            else:
                res["wilcoxon_p"] = None
            out["per_signed_flip"][sflip]["metrics"][m] = res

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump({"summary": out, "raw_pair_metrics": raw_pair_metrics}, f, indent=2)
    logger.info(f"Saved metrics -> {args.output_path}")

    # Concise log
    print("\n=== Per signed-flip ===")
    for sflip in ["+1", "-1", "0"]:
        s = out["per_signed_flip"].get(sflip, {})
        if not s.get("n"):
            continue
        print(f"\n{sflip} (toward {'more' if sflip == '+1' else 'less' if sflip == '-1' else 'same'} authority projection): n={s['n']}")
        for m, mv in s["metrics"].items():
            p_str = f"p={mv['wilcoxon_p']:.3f}" if mv.get("wilcoxon_p") is not None else "p=n/a"
            print(f"  {m:14s} mean_delta={mv['mean_delta']:+.3f} median={mv['median_delta']:+.2f}  {p_str}")


if __name__ == "__main__":
    main()
