"""Train linear probes on cached activations: layer sweep, Probe C (last-token) and optional Probe A (dense).

Run:
    uv run python -m auth_projection.train_probe
    uv run python -m auth_projection.train_probe --probe A   # requires dense activations
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_INPUT = Path(__file__).parent / "data" / "activations.pt"
DEFAULT_OUTPUT = Path(__file__).parent / "data" / "probe_results.json"
DEFAULT_PROBES_DIR = Path(__file__).parent / "data" / "probes"

LABEL_TO_INT = {"none": 0, "somewhat": 1, "strongly": 2}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


def split_by_seed(
    records: List[Dict], test_frac: float = 0.2, seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """Split records into train/test by seed_id (a conversation never crosses splits)."""
    rng = np.random.RandomState(seed)
    seed_ids = sorted({r["seed_id"] for r in records})
    rng.shuffle(seed_ids)
    n_test = max(1, int(len(seed_ids) * test_frac))
    test_seeds = set(seed_ids[:n_test])
    train = [r for r in records if r["seed_id"] not in test_seeds]
    test = [r for r in records if r["seed_id"] in test_seeds]
    return train, test


def build_probe_C(records: List[Dict], layer: int) -> Tuple[np.ndarray, np.ndarray]:
    """Probe C: one (last-user-token activation, turn label) pair per turn."""
    X = np.stack([r["last_token_act"][layer].float().numpy() for r in records])
    y = np.array([LABEL_TO_INT[r["label"]] for r in records])
    return X, y


def build_probe_A(records: List[Dict], layer: int) -> Tuple[np.ndarray, np.ndarray]:
    """Probe A: every user-turn token activation paired with the turn-level label.
    Requires `dense_act` saved in the records (run extract with --dense_layers including this layer).
    """
    Xs, ys = [], []
    for r in records:
        if "dense_act" not in r:
            continue
        try:
            local_idx = r["dense_layers"].index(layer)
        except ValueError:
            continue
        # dense_act shape: [n_dense_layers, n_tokens, hidden]
        token_acts = r["dense_act"][local_idx].float().numpy()  # [n_tokens, hidden]
        Xs.append(token_acts)
        ys.append(np.full(token_acts.shape[0], LABEL_TO_INT[r["label"]]))
    if not Xs:
        raise ValueError(
            f"No dense activations found at layer {layer}. "
            "Re-run extract_activations with --dense_layers including this layer."
        )
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y


def fit_eval(X_train, y_train, X_test, y_test) -> Dict:
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        multi_class="multinomial",
        solver="lbfgs",
    ).fit(X_train, y_train)

    probs = clf.predict_proba(X_test)
    preds = clf.predict(X_test)

    metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": float((preds == y_test).mean()),
    }
    for cls_name, cls_int in LABEL_TO_INT.items():
        binary_y = (y_test == cls_int).astype(int)
        if 0 < binary_y.sum() < len(binary_y):
            # Class index in clf may differ from LABEL_TO_INT if a class is absent in train
            if cls_int in clf.classes_:
                col = list(clf.classes_).index(cls_int)
                metrics[f"auc_{cls_name}_vs_rest"] = float(roc_auc_score(binary_y, probs[:, col]))
    return metrics, clf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--probes_dir", type=Path, default=DEFAULT_PROBES_DIR)
    parser.add_argument("--probe", choices=["C", "A"], default="C",
                        help="C = last-user-token (default); A = dense per-token (needs dense activations).")
    parser.add_argument("--layers", default="",
                        help="Comma-separated layers to train probes at. Empty = all layers (sweep).")
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_clfs", action="store_true",
                        help="Save sklearn LogisticRegression objects per layer (joblib).")
    args = parser.parse_args()

    records = torch.load(args.input_path, map_location="cpu")
    n_layers_plus1 = records[0]["last_token_act"].shape[0]
    logger.info(f"Loaded {len(records)} records | n_layers+1 = {n_layers_plus1}")
    logger.info(f"Label dist: {dict(Counter(r['label'] for r in records))}")

    train_recs, test_recs = split_by_seed(records, args.test_frac, args.seed)
    logger.info(f"Train: {len(train_recs)} | Test: {len(test_recs)}")

    layers = (
        [int(x) for x in args.layers.split(",") if x.strip()]
        if args.layers
        else list(range(n_layers_plus1))
    )

    builder = build_probe_C if args.probe == "C" else build_probe_A

    results = {"probe": args.probe, "n_train_convs": len({r["seed_id"] for r in train_recs}),
               "n_test_convs": len({r["seed_id"] for r in test_recs}), "per_layer": {}}

    args.probes_dir.mkdir(parents=True, exist_ok=True)
    for layer in layers:
        try:
            X_train, y_train = builder(train_recs, layer)
            X_test, y_test = builder(test_recs, layer)
        except ValueError as e:
            logger.warning(f"Layer {layer}: skipping ({e})")
            continue

        metrics, clf = fit_eval(X_train, y_train, X_test, y_test)
        results["per_layer"][str(layer)] = metrics
        logger.info(
            f"Layer {layer:2d} | acc={metrics['accuracy']:.3f} | "
            + " ".join(f"auc_{k}={v:.3f}" for k, v in metrics.items() if k.startswith("auc_"))
        )

        if args.save_clfs:
            import joblib
            joblib.dump(clf, args.probes_dir / f"probe_{args.probe}_layer{layer}.joblib")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved metrics to {args.output_path}")

    # Quick summary of best layer by `auc_strongly_vs_rest` (or accuracy as fallback)
    def score(L):
        return results["per_layer"][L].get("auc_strongly_vs_rest", results["per_layer"][L]["accuracy"])
    best = max(results["per_layer"].keys(), key=score)
    logger.info(f"Best layer: {best} | metrics: {results['per_layer'][best]}")


if __name__ == "__main__":
    main()
