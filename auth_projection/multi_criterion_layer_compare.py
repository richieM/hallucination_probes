"""For each model version (v1, v2), evaluate every layer's saved probe on the standard
test set and counterfactuals, then report the layer chosen by 4 different criteria.

Criteria:
  - max accuracy on standard test
  - max AUC strongly-vs-rest on standard test
  - min CF gap on AUC strongly-vs-rest (smaller = more paraphrase-robust)
  - max minimal-edit flip-correct rate

If the criteria pick the same layer, layer selection isn't biasing us. If they disagree
wildly, the disagreement itself is informative and we should report results at multiple
layers in the writeup.

Run:
    uv run python -m auth_projection.multi_criterion_layer_compare
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import torch

from .train_probe import LABEL_TO_INT, INT_TO_LABEL, build_probe_C, split_by_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def auc_one_vs_rest(probs, y, classes_, label_int):
    from sklearn.metrics import roc_auc_score
    bin_y = (y == label_int).astype(int)
    if bin_y.sum() == 0 or bin_y.sum() == len(bin_y) or label_int not in classes_:
        return None
    col = list(classes_).index(label_int)
    return float(roc_auc_score(bin_y, probs[:, col]))


def eval_layer(layer: int, probes_dir: Path, train_records, test_records,
               para_records, mined_records, full_records):
    """Return metrics dict for one layer at the saved probe."""
    probe_path = probes_dir / f"probe_C_layer{layer}.joblib"
    if not probe_path.exists():
        return None
    clf = joblib.load(probe_path)

    X_test, y_test = build_probe_C(test_records, layer)
    probs = clf.predict_proba(X_test)
    preds = clf.predict(X_test)
    acc = float((preds == y_test).mean())
    auc_str_test = auc_one_vs_rest(probs, y_test, clf.classes_, LABEL_TO_INT["strongly"])

    # Paraphrase
    auc_str_para = None
    if para_records:
        try:
            X_p, y_p = build_probe_C(para_records, layer)
            probs_p = clf.predict_proba(X_p)
            auc_str_para = auc_one_vs_rest(probs_p, y_p, clf.classes_, LABEL_TO_INT["strongly"])
        except Exception:
            pass
    cf_gap = (auc_str_test - auc_str_para) if (auc_str_test is not None and auc_str_para is not None) else None

    # Minimal-edit flip rate
    flip_rate = None
    if mined_records and full_records:
        # For each minedit child, find its parent by seed
        full_idx = {(r["seed_id"], r["turn_index"]): r for r in full_records}
        flip_attempted = 0
        flip_correct = 0
        for c in mined_records:
            if "__minedit_to_" not in c["seed_id"]:
                continue
            parent_seed = c["seed_id"].split("__minedit_to_")[0]
            p = full_idx.get((parent_seed, c["turn_index"]))
            if p is None or p["label"] == c["label"]:
                continue
            flip_attempted += 1
            X_p = p["last_token_act"][layer].float().numpy()[None, :]
            X_c = c["last_token_act"][layer].float().numpy()[None, :]
            pred_p = clf.predict(X_p)[0]
            pred_c = clf.predict(X_c)[0]
            if pred_p != pred_c:
                if pred_p == LABEL_TO_INT[p["label"]] and pred_c == LABEL_TO_INT[c["label"]]:
                    flip_correct += 1
        if flip_attempted > 0:
            flip_rate = flip_correct / flip_attempted

    return {
        "layer": layer,
        "accuracy": acc,
        "auc_strongly_test": auc_str_test,
        "auc_strongly_paraphrase": auc_str_para,
        "cf_gap_strongly": cf_gap,
        "flip_attempted": flip_attempted if mined_records else None,
        "flip_correct_rate": flip_rate,
    }


def run_for_version(version: str, data_dir: Path, n_layers_plus1: int):
    activations = data_dir / f"{version}_activations.pt"
    para = data_dir / f"{version}_cf_paraphrase_activations.pt"
    mined = data_dir / f"{version}_cf_minimal_edit_activations.pt"
    probes = data_dir / f"{version}_probes"
    if not activations.exists() or not probes.exists():
        logger.warning(f"{version}: missing activations or probes; skipping")
        return None

    logger.info(f"=== {version} ===")
    records = torch.load(activations, map_location="cpu")
    train_recs, test_recs = split_by_seed(records, test_frac=0.2, seed=42)
    para_recs = torch.load(para, map_location="cpu") if para.exists() else None
    mined_recs = torch.load(mined, map_location="cpu") if mined.exists() else None

    per_layer = []
    for L in range(n_layers_plus1):
        m = eval_layer(L, probes, train_recs, test_recs, para_recs, mined_recs, records)
        if m:
            per_layer.append(m)
    return per_layer


def best_by(per_layer: List[Dict], key: str, direction: str = "max"):
    """direction='max' or 'min'. Skips entries where key is None."""
    valid = [m for m in per_layer if m.get(key) is not None]
    if not valid:
        return None
    if direction == "max":
        return max(valid, key=lambda m: m[key])
    return min(valid, key=lambda m: m[key])


def main():
    data_dir = Path("auth_projection/data")
    out = {}
    for v, n_layers_plus1 in [("v1", 29), ("v2", 37)]:
        per = run_for_version(v, data_dir, n_layers_plus1)
        if not per:
            continue
        bests = {
            "max_accuracy": best_by(per, "accuracy", "max"),
            "max_auc_strongly": best_by(per, "auc_strongly_test", "max"),
            "min_cf_gap": best_by(per, "cf_gap_strongly", "min"),
            "max_flip_rate": best_by(per, "flip_correct_rate", "max"),
        }
        out[v] = {"per_layer": per, "bests": bests}

        print(f"\n=== {v} — best layer by criterion ===")
        for crit, m in bests.items():
            if m is None:
                print(f"  {crit:<20} (n/a)")
                continue
            print(f"  {crit:<20} layer={m['layer']:<3} acc={m['accuracy']:.3f} AUC_str={m['auc_strongly_test']:.3f} "
                  f"CF_gap={m['cf_gap_strongly'] if m['cf_gap_strongly'] is not None else 'n/a':<6} "
                  f"flip%={m['flip_correct_rate']*100 if m['flip_correct_rate'] is not None else float('nan'):.1f}")

        chosen_layers = sorted({m["layer"] for m in bests.values() if m})
        print(f"  layers picked: {chosen_layers}")

    with open(data_dir / "multi_criterion_layer_compare.json", "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved -> {data_dir / 'multi_criterion_layer_compare.json'}")


if __name__ == "__main__":
    main()
