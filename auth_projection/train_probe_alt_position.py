"""B1: train probes on the assistant-start-token activations (instead of last-user-token).

Reads the `assistant_start_token_act` field from saved activations (which we
added in extract_activations.py). Otherwise mirrors train_probe.py: layer sweep,
balanced LR, seed-stratified train/test split.

Run:
    uv run python -m auth_projection.train_probe_alt_position \\
        --input_path auth_projection/data/v3b_activations.pt \\
        --output_path auth_projection/data/v3b_probe_results_assistant_start.json \\
        --probes_dir auth_projection/data/v3b_probes_assistant_start \\
        --save_clfs
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .train_probe import LABEL_TO_INT, INT_TO_LABEL, fit_eval, split_by_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_probe_alt(records: List[Dict], layer: int) -> Tuple[np.ndarray, np.ndarray]:
    """Like build_probe_C but reads assistant_start_token_act."""
    Xs, ys = [], []
    for r in records:
        if "assistant_start_token_act" not in r:
            continue
        Xs.append(r["assistant_start_token_act"][layer].float().numpy())
        ys.append(LABEL_TO_INT[r["label"]])
    if not Xs:
        raise ValueError(f"No assistant_start_token_act records found")
    return np.stack(Xs), np.array(ys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--probes_dir", type=Path, required=True)
    parser.add_argument("--save_clfs", action="store_true")
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = torch.load(args.input_path, map_location="cpu")
    n_with_alt = sum(1 for r in records if "assistant_start_token_act" in r)
    logger.info(f"Loaded {len(records)} records | with assistant_start_token_act: {n_with_alt}")

    if n_with_alt == 0:
        raise SystemExit("No records have assistant_start_token_act. "
                         "Re-extract activations with the patched extract_activations.py.")

    n_layers_plus1 = records[0]["assistant_start_token_act"].shape[0] \
        if "assistant_start_token_act" in records[0] else None
    if n_layers_plus1 is None:
        for r in records:
            if "assistant_start_token_act" in r:
                n_layers_plus1 = r["assistant_start_token_act"].shape[0]
                break

    logger.info(f"n_layers+1 = {n_layers_plus1}")
    train_recs, test_recs = split_by_seed(records, args.test_frac, args.seed)
    logger.info(f"Train: {len(train_recs)} | Test: {len(test_recs)}")
    logger.info(f"Label dist: {dict(Counter(r['label'] for r in records))}")

    args.probes_dir.mkdir(parents=True, exist_ok=True)
    results = {"probe_position": "assistant_start_token",
               "n_train_convs": len({r['seed_id'] for r in train_recs}),
               "n_test_convs": len({r['seed_id'] for r in test_recs}),
               "per_layer": {}}

    for layer in range(n_layers_plus1):
        try:
            X_train, y_train = build_probe_alt(train_recs, layer)
            X_test, y_test = build_probe_alt(test_recs, layer)
        except ValueError as e:
            logger.warning(f"Layer {layer}: {e}")
            continue
        metrics, clf = fit_eval(X_train, y_train, X_test, y_test)
        results["per_layer"][str(layer)] = metrics
        logger.info(f"Layer {layer:2d} | acc={metrics['accuracy']:.3f} | "
                    + " ".join(f"{k}={v:.3f}" for k, v in metrics.items() if k.startswith("auc_")))
        if args.save_clfs:
            joblib.dump(clf, args.probes_dir / f"probe_C_layer{layer}.joblib")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved -> {args.output_path}")

    # Headline summary by acc and AUC strongly
    sorted_acc = sorted(results["per_layer"].items(), key=lambda kv: kv[1]["accuracy"], reverse=True)
    sorted_auc = sorted(results["per_layer"].items(),
                        key=lambda kv: kv[1].get("auc_strongly_vs_rest", 0), reverse=True)
    print("\nTop 5 layers by accuracy (assistant-start-token probe):")
    for L, m in sorted_acc[:5]:
        print(f"  L{L}: acc={m['accuracy']:.3f} AUC_str={m.get('auc_strongly_vs_rest', 0):.3f}")
    print("Top 5 layers by AUC strongly:")
    for L, m in sorted_auc[:5]:
        print(f"  L{L}: acc={m['accuracy']:.3f} AUC_str={m.get('auc_strongly_vs_rest', 0):.3f}")


if __name__ == "__main__":
    main()
