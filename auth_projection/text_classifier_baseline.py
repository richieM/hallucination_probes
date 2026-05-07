"""Text-only classifier baselines, mirroring the probe's eval setup exactly.

For each labeled user turn, encode the conversation prefix through that turn as text and
train a balanced LR with the same train/test seed split as the probe. Evaluate on:
  - standard held-out test
  - paraphrase counterfactuals (CF gap)
  - minimal-edit pairs (flip-correct rate)

Two encoders:
  - TF-IDF (cheapest sanity baseline)
  - sentence-transformers all-MiniLM-L6-v2 (sensible blackbox encoder, runs on CPU/MPS)

Probe @ L17 numbers to beat:
  acc=0.774 | AUC strongly=0.975 | paraphrase AUC strongly=0.954 (CF gap 0.021)
  | minimal-edit flip-correct rate=40.4% on 280 attempted flips

Run:
    uv run python -m auth_projection.text_classifier_baseline
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from utils.file_utils import load_jsonl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


LABEL_TO_INT = {"none": 0, "somewhat": 1, "strongly": 2}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


def conv_prefix_text(conversation: List[Dict], user_turn_index: int) -> str:
    out = []
    user_count = 0
    for t in conversation:
        out.append(f"{t['role'].upper()}: {t['content']}")
        if t["role"] == "user":
            if user_count == user_turn_index:
                return "\n\n".join(out)
            user_count += 1
    raise ValueError(f"user_turn_index={user_turn_index} not found")


def conversations_to_records(rows: List[Dict]) -> List[Dict]:
    out = []
    for conv in rows:
        labels = conv.get("turn_labels") or []
        for tl in labels:
            try:
                prefix = conv_prefix_text(conv["conversation"], tl["turn_index"])
            except ValueError:
                continue
            out.append({
                "seed_id": conv["seed_id"],
                "turn_index": tl["turn_index"],
                "label": tl["label"],
                "prefix_text": prefix,
                "user_turn_text": conv["conversation"][2 * tl["turn_index"]]["content"],
            })
    return out


def split_by_seed(records: List[Dict], test_frac: float = 0.2, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    rng = np.random.RandomState(seed)
    seed_ids = sorted({r["seed_id"] for r in records})
    rng.shuffle(seed_ids)
    n_test = max(1, int(len(seed_ids) * test_frac))
    test_seeds = set(seed_ids[:n_test])
    return [r for r in records if r["seed_id"] not in test_seeds], [r for r in records if r["seed_id"] in test_seeds]


def featurize_tfidf(train_texts: List[str], test_texts: List[str], **other_texts) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], TfidfVectorizer]:
    vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_features=20000)
    X_train = vec.fit_transform(train_texts).toarray()
    X_test = vec.transform(test_texts).toarray()
    others = {k: vec.transform(v).toarray() for k, v in other_texts.items()}
    return X_train, X_test, others, vec


def featurize_st(train_texts: List[str], test_texts: List[str], **other_texts) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], object]:
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    X_train = enc.encode(train_texts, convert_to_numpy=True, show_progress_bar=True, batch_size=32)
    X_test = enc.encode(test_texts, convert_to_numpy=True, show_progress_bar=True, batch_size=32)
    others = {k: enc.encode(v, convert_to_numpy=True, show_progress_bar=False, batch_size=32) for k, v in other_texts.items()}
    return X_train, X_test, others, enc


def fit_eval(X_train, y_train, X_test, y_test) -> Tuple[Dict, LogisticRegression]:
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", n_jobs=-1).fit(X_train, y_train)
    probs = clf.predict_proba(X_test)
    preds = clf.predict(X_test)
    metrics: Dict[str, float] = {
        "n": int(len(X_test)),
        "accuracy": float((preds == y_test).mean()),
    }
    for cls_name, cls_int in LABEL_TO_INT.items():
        bin_y = (y_test == cls_int).astype(int)
        if 0 < bin_y.sum() < len(bin_y) and cls_int in clf.classes_:
            col = list(clf.classes_).index(cls_int)
            metrics[f"auc_{cls_name}_vs_rest"] = float(roc_auc_score(bin_y, probs[:, col]))
    return metrics, clf


def minimal_edit_flip_rate(clf, parent_records: List[Dict], child_records: List[Dict],
                           parent_X: np.ndarray, child_X: np.ndarray) -> Dict:
    """Mirror eval_counterfactuals.eval_minimal_edit_pairs but on text features.
    parent_records list aligns with parent_X rows; same for child."""
    parent_X_by_key = {(r["seed_id"], r["turn_index"]): parent_X[i] for i, r in enumerate(parent_records)}
    parent_label_by_key = {(r["seed_id"], r["turn_index"]): r["label"] for r in parent_records}

    pairs_checked = 0
    flip_attempted = 0
    flip_correct = 0
    for i, c in enumerate(child_records):
        if "__minedit_to_" not in c["seed_id"]:
            continue
        parent_seed = c["seed_id"].split("__minedit_to_")[0]
        key = (parent_seed, c["turn_index"])
        if key not in parent_X_by_key:
            continue
        pairs_checked += 1
        if parent_label_by_key[key] == c["label"]:
            continue
        flip_attempted += 1
        pred_p = clf.predict(parent_X_by_key[key][None, :])[0]
        pred_c = clf.predict(child_X[i][None, :])[0]
        if INT_TO_LABEL[pred_p] != INT_TO_LABEL[pred_c]:
            if (pred_p == LABEL_TO_INT[parent_label_by_key[key]]) and (pred_c == LABEL_TO_INT[c["label"]]):
                flip_correct += 1
    return {
        "pairs_checked": pairs_checked,
        "flip_attempted": flip_attempted,
        "flip_correct": flip_correct,
        "flip_correct_rate": float(flip_correct / max(1, flip_attempted)),
    }


def run_one_encoder(name: str, train_recs, test_recs, para_recs, mined_recs, full_train_recs, encoder_fn) -> Dict:
    logger.info(f"\n=== Encoder: {name} ===")
    train_texts = [r["prefix_text"] for r in train_recs]
    test_texts = [r["prefix_text"] for r in test_recs]
    para_texts = [r["prefix_text"] for r in para_recs]
    mined_texts = [r["prefix_text"] for r in mined_recs]
    full_train_texts = [r["prefix_text"] for r in full_train_recs]

    X_train, X_test, others, _ = encoder_fn(
        train_texts, test_texts,
        para=para_texts, mined=mined_texts, full_train=full_train_texts,
    )
    y_train = np.array([LABEL_TO_INT[r["label"]] for r in train_recs])
    y_test = np.array([LABEL_TO_INT[r["label"]] for r in test_recs])
    y_para = np.array([LABEL_TO_INT[r["label"]] for r in para_recs])

    standard, clf = fit_eval(X_train, y_train, X_test, y_test)

    para_metrics = {"n": int(len(para_texts))}
    if len(para_texts):
        para_probs = clf.predict_proba(others["para"])
        para_preds = clf.predict(others["para"])
        para_metrics["accuracy"] = float((para_preds == y_para).mean())
        for cls_name, cls_int in LABEL_TO_INT.items():
            bin_y = (y_para == cls_int).astype(int)
            if 0 < bin_y.sum() < len(bin_y) and cls_int in clf.classes_:
                col = list(clf.classes_).index(cls_int)
                para_metrics[f"auc_{cls_name}_vs_rest"] = float(roc_auc_score(bin_y, para_probs[:, col]))

    cf_gap = {}
    for k in standard:
        if k.startswith("auc_") and k in para_metrics:
            cf_gap[k] = float(standard[k] - para_metrics[k])

    minedit = minimal_edit_flip_rate(clf, full_train_recs, mined_recs, others["full_train"], others["mined"])

    return {
        "encoder": name,
        "standard_test": standard,
        "paraphrase": para_metrics,
        "counterfactual_gap": cf_gap,
        "minimal_edit": minedit,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_path", type=Path, default=Path("auth_projection/data/v1_labeled.jsonl"))
    parser.add_argument("--paraphrase_path", type=Path, default=Path("auth_projection/data/v1_cf_paraphrase_labeled.jsonl"))
    parser.add_argument("--minedit_path", type=Path, default=Path("auth_projection/data/v1_cf_minimal_edit_labeled.jsonl"))
    parser.add_argument("--output_path", type=Path, default=Path("auth_projection/data/v1_text_classifier_results.json"))
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main_recs = conversations_to_records(load_jsonl(args.main_path))
    para_recs = conversations_to_records(load_jsonl(args.paraphrase_path))
    mined_recs = conversations_to_records(load_jsonl(args.minedit_path))
    logger.info(f"Records: main={len(main_recs)} para={len(para_recs)} mined={len(mined_recs)}")

    train_recs, test_recs = split_by_seed(main_recs, args.test_frac, args.seed)
    logger.info(f"Train={len(train_recs)} Test={len(test_recs)}")
    logger.info(f"Train label dist: {Counter(r['label'] for r in train_recs)}")
    logger.info(f"Test  label dist: {Counter(r['label'] for r in test_recs)}")

    results = {"probe_baseline_layer17": {
        "standard_test": {"accuracy": 0.774, "auc_strongly_vs_rest": 0.975, "auc_none_vs_rest": 0.881, "auc_somewhat_vs_rest": 0.772},
        "paraphrase":   {"accuracy": 0.712, "auc_strongly_vs_rest": 0.954, "auc_none_vs_rest": 0.847, "auc_somewhat_vs_rest": 0.709},
        "counterfactual_gap": {"auc_strongly_vs_rest": 0.021, "auc_none_vs_rest": 0.033, "auc_somewhat_vs_rest": 0.063},
        "minimal_edit": {"flip_correct_rate": 0.404, "flip_attempted": 280},
    }, "encoders": {}}

    # full_train_recs is used for minimal-edit parent lookup (parents may be in train OR test split)
    full_train_recs = main_recs

    # Also build user-turn-only views for fairness against the 512-token encoder
    def user_turn_only(recs):
        return [{**r, "prefix_text": r["user_turn_text"]} for r in recs]

    train_uto = user_turn_only(train_recs)
    test_uto = user_turn_only(test_recs)
    para_uto = user_turn_only(para_recs)
    mined_uto = user_turn_only(mined_recs)
    full_train_uto = user_turn_only(full_train_recs)

    runs = [
        ("tfidf_full_prefix", featurize_tfidf, train_recs, test_recs, para_recs, mined_recs, full_train_recs),
        ("tfidf_user_turn_only", featurize_tfidf, train_uto, test_uto, para_uto, mined_uto, full_train_uto),
        ("st_full_prefix", featurize_st, train_recs, test_recs, para_recs, mined_recs, full_train_recs),
        ("st_user_turn_only", featurize_st, train_uto, test_uto, para_uto, mined_uto, full_train_uto),
    ]
    for name, fn, tr, te, pa, mi, ft in runs:
        try:
            results["encoders"][name] = run_one_encoder(name, tr, te, pa, mi, ft, fn)
        except Exception as e:
            logger.exception(f"{name} failed: {e}")
            results["encoders"][name] = {"error": str(e)}

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved -> {args.output_path}")

    # Summary print
    print("\n=== SUMMARY ===")
    print(f"{'encoder':<28} {'acc':<6} {'AUC_str':<8} {'CF_gap_str':<10} {'minedit_flip%':<12}")
    pb = results["probe_baseline_layer17"]
    print(f"{'probe_L17 (residual)':<28} {pb['standard_test']['accuracy']:<6.3f} {pb['standard_test']['auc_strongly_vs_rest']:<8.3f} {pb['counterfactual_gap']['auc_strongly_vs_rest']:<10.3f} {pb['minimal_edit']['flip_correct_rate']*100:<12.1f}")
    for name, r in results["encoders"].items():
        if "error" in r:
            print(f"{name:<28} ERROR: {r['error']}")
            continue
        s = r["standard_test"]
        gap = r["counterfactual_gap"].get("auc_strongly_vs_rest", float("nan"))
        flip = r["minimal_edit"]["flip_correct_rate"] * 100
        print(f"{name:<28} {s['accuracy']:<6.3f} {s.get('auc_strongly_vs_rest', float('nan')):<8.3f} {gap:<10.3f} {flip:<12.1f}")


if __name__ == "__main__":
    main()
