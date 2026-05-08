"""Evaluate trained probes on counterfactual eval sets.

Loads counterfactual activations + the trained probe at the best layer, computes:
  - Standard test AUC (from train_probe.py)
  - Paraphrase AUC: same probe, on activations from paraphrased convs
  - Minimal-edit pair flip rate: does the probe label flip when the conv label flipped?

Counterfactual_gap = standard_AUC - paraphrase_AUC. Small gap = probe captures state, not surface.

Run:
    # Step 1: extract activations for counterfactual convs (after labeling them)
    uv run python -m auth_projection.extract_activations \
        --input_path auth_projection/data/counterfactual_paraphrase_labeled.jsonl \
        --output_path auth_projection/data/activations_paraphrase.pt
    uv run python -m auth_projection.extract_activations \
        --input_path auth_projection/data/counterfactual_minimal_edit_labeled.jsonl \
        --output_path auth_projection/data/activations_minimal_edit.pt

    # Step 2: eval
    uv run python -m auth_projection.eval_counterfactuals --layer 24
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .train_probe import (
    DEFAULT_PROBES_DIR,
    LABEL_TO_INT,
    INT_TO_LABEL,
    build_probe_C,
    fit_eval,
    split_by_seed,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_TRAIN_ACTIVATIONS = Path(__file__).parent / "data" / "activations.pt"
DEFAULT_PARAPHRASE_ACTIVATIONS = Path(__file__).parent / "data" / "activations_paraphrase.pt"
DEFAULT_MINEDIT_ACTIVATIONS = Path(__file__).parent / "data" / "activations_minimal_edit.pt"
DEFAULT_OUTPUT = Path(__file__).parent / "data" / "counterfactual_eval.json"


def load_or_train_probe(
    train_activations_path: Path,
    layer: int,
    probes_dir: Path,
    test_frac: float,
    seed: int,
):
    """Load saved probe if available, else retrain on the train split at the given layer."""
    saved = probes_dir / f"probe_C_layer{layer}.joblib"
    if saved.exists():
        logger.info(f"Loading saved probe: {saved}")
        return joblib.load(saved)

    logger.info(f"No saved probe at layer {layer}; retraining from {train_activations_path}")
    records = torch.load(train_activations_path, map_location="cpu")
    train_recs, _ = split_by_seed(records, test_frac=test_frac, seed=seed)
    X_train, y_train = build_probe_C(train_recs, layer)
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(
        max_iter=2000, class_weight="balanced", solver="lbfgs"
    ).fit(X_train, y_train)
    probes_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, saved)
    return clf


def eval_paraphrase(
    clf, layer: int, paraphrase_records: List[Dict]
) -> Dict:
    """Compute AUC + accuracy on paraphrased convs (label preserved from labeler)."""
    X, y = build_probe_C(paraphrase_records, layer)
    probs = clf.predict_proba(X)
    preds = clf.predict(X)
    metrics = {"n": len(X), "accuracy": float((preds == y).mean())}
    for cls_name, cls_int in LABEL_TO_INT.items():
        bin_y = (y == cls_int).astype(int)
        if 0 < bin_y.sum() < len(bin_y) and cls_int in clf.classes_:
            col = list(clf.classes_).index(cls_int)
            metrics[f"auc_{cls_name}_vs_rest"] = float(roc_auc_score(bin_y, probs[:, col]))
    return metrics


def eval_minimal_edit_pairs(
    clf,
    layer: int,
    minedit_records: List[Dict],
    train_records: List[Dict],
) -> Dict:
    """For each minimal-edit child whose parent is in the held-out training set,
    check whether the probe flips its label between (parent, child).
    Each minedit_record has seed_id like '<parent_seed>__minedit_to_<flip>'."""
    parent_by_id = {r["seed_id"]: r for r in train_records}

    pairs_checked = 0
    flip_correct = 0  # probe's label changes in the same direction as the labeler's
    flip_attempted = 0  # cases where the labeler's label actually changed

    for child in minedit_records:
        seed = child["seed_id"]
        if "__minedit_to_" not in seed:
            continue
        parent_seed = seed.split("__minedit_to_")[0]
        # Collect all parent-turn records and child-turn records
        parent_turns = [r for r in train_records if r["seed_id"] == parent_seed]
        child_turns = [r for r in minedit_records if r["seed_id"] == seed]
        # Pair by turn_index
        parent_by_turn = {r["turn_index"]: r for r in parent_turns}
        for c in child_turns:
            p = parent_by_turn.get(c["turn_index"])
            if p is None:
                continue
            pairs_checked += 1
            if p["label"] != c["label"]:
                flip_attempted += 1
                pred_p = clf.predict(p["last_token_act"][layer].float().numpy()[None, :])[0]
                pred_c = clf.predict(c["last_token_act"][layer].float().numpy()[None, :])[0]
                if INT_TO_LABEL[pred_p] != INT_TO_LABEL[pred_c]:
                    # Probe flipped; check direction matches
                    parent_label_int = LABEL_TO_INT[p["label"]]
                    child_label_int = LABEL_TO_INT[c["label"]]
                    if (pred_p == parent_label_int) and (pred_c == child_label_int):
                        flip_correct += 1
    return {
        "pairs_checked": pairs_checked,
        "flip_attempted": flip_attempted,
        "flip_correct": flip_correct,
        "flip_correct_rate": float(flip_correct / max(1, flip_attempted)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True, help="Layer at which to eval (typically the best layer from train_probe.py).")
    parser.add_argument("--train_activations", type=Path, default=DEFAULT_TRAIN_ACTIVATIONS)
    parser.add_argument("--paraphrase_activations", type=Path, default=DEFAULT_PARAPHRASE_ACTIVATIONS)
    parser.add_argument("--minedit_activations", type=Path, default=DEFAULT_MINEDIT_ACTIVATIONS)
    parser.add_argument("--probes_dir", type=Path, default=DEFAULT_PROBES_DIR)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    clf = load_or_train_probe(
        args.train_activations, args.layer, args.probes_dir, args.test_frac, args.seed
    )

    train_records = torch.load(args.train_activations, map_location="cpu")
    train_recs, test_recs = split_by_seed(train_records, args.test_frac, args.seed)

    # Holdout assertion: counterfactual data must derive from held-out test seeds, not train seeds.
    train_seed_set = {r["seed_id"] for r in train_recs}
    if args.paraphrase_activations.exists():
        para_recs_check = torch.load(args.paraphrase_activations, map_location="cpu")
        para_parent_seeds = {r["seed_id"].split("__")[0] for r in para_recs_check}
        leaked = para_parent_seeds & train_seed_set
        assert not leaked, f"paraphrase CF leaks train seeds: {leaked}"
    if args.minedit_activations.exists():
        minedit_recs_check = torch.load(args.minedit_activations, map_location="cpu")
        minedit_parent_seeds = {r["seed_id"].split("__minedit_to_")[0] for r in minedit_recs_check
                                 if "__minedit_to_" in r["seed_id"]}
        leaked = minedit_parent_seeds & train_seed_set
        assert not leaked, f"minimal-edit CF leaks train seeds: {leaked}"

    # Standard test AUC at this layer
    X_test, y_test = build_probe_C(test_recs, args.layer)
    probs = clf.predict_proba(X_test)
    standard = {"n": int(len(X_test)), "accuracy": float((clf.predict(X_test) == y_test).mean())}
    for cls_name, cls_int in LABEL_TO_INT.items():
        bin_y = (y_test == cls_int).astype(int)
        if 0 < bin_y.sum() < len(bin_y) and cls_int in clf.classes_:
            col = list(clf.classes_).index(cls_int)
            standard[f"auc_{cls_name}_vs_rest"] = float(roc_auc_score(bin_y, probs[:, col]))

    out: Dict = {"layer": args.layer, "standard_test": standard}

    if args.paraphrase_activations.exists():
        para_recs = torch.load(args.paraphrase_activations, map_location="cpu")
        out["paraphrase"] = eval_paraphrase(clf, args.layer, para_recs)
        # Counterfactual gap per class
        out["counterfactual_gap"] = {}
        for k in standard:
            if k.startswith("auc_") and k in out["paraphrase"]:
                out["counterfactual_gap"][k] = float(standard[k] - out["paraphrase"][k])
    else:
        logger.warning(f"No paraphrase activations at {args.paraphrase_activations}")

    if args.minedit_activations.exists():
        minedit_recs = torch.load(args.minedit_activations, map_location="cpu")
        out["minimal_edit"] = eval_minimal_edit_pairs(clf, args.layer, minedit_recs, train_records)
    else:
        logger.warning(f"No minimal-edit activations at {args.minedit_activations}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved counterfactual eval to {args.output_path}")
    logger.info(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
