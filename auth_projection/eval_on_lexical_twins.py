"""Compare probe (L17, residual-stream) vs TF-IDF (user-turn-only) on the lexical-twin
subset of the held-out test set.

Lexical-twin convs are the ones where the data design specifically tried to break
lexical detection:
  - peer_voice: target_tier=strongly but user uses peer-like vocabulary, no submission language
  - submission_voice: target_tier=none but uses deferential-sounding language without behavioral surrender

If the probe has a real "captures behavioral state, not just words" advantage, it should
win on this subset. If TF-IDF tracks the probe even here, the lexical-twin design
protected against one specific leak but not the broader lexical correlation.

Run:
    uv run python -m auth_projection.eval_on_lexical_twins
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from utils.file_utils import load_jsonl

from .train_probe import LABEL_TO_INT, INT_TO_LABEL, build_probe_C, split_by_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def auc_one_vs_rest(probs, y, classes_, label_int):
    bin_y = (y == label_int).astype(int)
    if bin_y.sum() == 0 or bin_y.sum() == len(bin_y):
        return None
    if label_int not in classes_:
        return None
    col = list(classes_).index(label_int)
    return float(roc_auc_score(bin_y, probs[:, col]))


def main():
    LAYER = 17
    activations_path = Path("auth_projection/data/v1_activations.pt")
    labeled_path = Path("auth_projection/data/v1_labeled.jsonl")
    probe_path = Path("auth_projection/data/v1_probes/probe_C_layer17.joblib")

    records = torch.load(activations_path, map_location="cpu")
    train_recs, test_recs = split_by_seed(records, test_frac=0.2, seed=42)
    logger.info(f"Train={len(train_recs)} Test={len(test_recs)}")

    convs = load_jsonl(labeled_path)
    twin_by_seed = {c["seed_id"]: c.get("lexical_twin_kind") for c in convs}
    user_text_by_key = {}
    for c in convs:
        ut = 0
        for t in c["conversation"]:
            if t["role"] == "user":
                user_text_by_key[(c["seed_id"], ut)] = t["content"]
                ut += 1

    def attach(recs):
        out = []
        for r in recs:
            r2 = dict(r)
            r2["lexical_twin_kind"] = twin_by_seed.get(r["seed_id"])
            r2["user_turn_text"] = user_text_by_key.get((r["seed_id"], r["turn_index"]))
            out.append(r2)
        return out
    train_recs = attach(train_recs)
    test_recs = attach(test_recs)

    # ---- Probe at L17 ----
    clf_probe: LogisticRegression = joblib.load(probe_path)
    X_test, y_test = build_probe_C(test_recs, LAYER)
    probe_probs = clf_probe.predict_proba(X_test)
    probe_preds = clf_probe.predict(X_test)

    # ---- TF-IDF on user-turn-only ----
    vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_features=20000)
    X_train_tfidf = vec.fit_transform([r["user_turn_text"] for r in train_recs]).toarray()
    X_test_tfidf = vec.transform([r["user_turn_text"] for r in test_recs]).toarray()
    y_train = np.array([LABEL_TO_INT[r["label"]] for r in train_recs])
    clf_tfidf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", n_jobs=-1).fit(X_train_tfidf, y_train)
    tfidf_probs = clf_tfidf.predict_proba(X_test_tfidf)
    tfidf_preds = clf_tfidf.predict(X_test_tfidf)

    # ---- Slice by lexical_twin_kind ----
    test_arr = np.arange(len(test_recs))
    slices = defaultdict(list)
    for i, r in enumerate(test_recs):
        slices["all"].append(i)
        slices[r.get("lexical_twin_kind") or "match"].append(i)
        if (r.get("lexical_twin_kind") in ("peer_voice", "submission_voice")):
            slices["lexical_twin_combined"].append(i)

    out = {}
    for slc_name, idx_list in slices.items():
        idx = np.array(idx_list)
        if len(idx) == 0:
            continue
        y_slc = y_test[idx]
        n = len(idx)
        if (y_slc == LABEL_TO_INT["strongly"]).sum() == 0 or (y_slc == LABEL_TO_INT["strongly"]).sum() == n:
            auc_p = auc_t = None
        else:
            auc_p = auc_one_vs_rest(probe_probs[idx], y_slc, clf_probe.classes_, LABEL_TO_INT["strongly"])
            auc_t = auc_one_vs_rest(tfidf_probs[idx], y_slc, clf_tfidf.classes_, LABEL_TO_INT["strongly"])
        acc_p = float((probe_preds[idx] == y_slc).mean())
        acc_t = float((tfidf_preds[idx] == y_slc).mean())
        out[slc_name] = {
            "n": int(n),
            "label_dist": {INT_TO_LABEL[l]: int((y_slc == l).sum()) for l in [0, 1, 2]},
            "probe_acc": acc_p,
            "tfidf_acc": acc_t,
            "probe_auc_strongly_vs_rest": auc_p,
            "tfidf_auc_strongly_vs_rest": auc_t,
        }

    # Print
    print(f"\n{'slice':<24} {'n':<4} {'labels (n/sw/st)':<18} {'probe_acc':<10} {'tfidf_acc':<10} {'probe_AUC_str':<14} {'tfidf_AUC_str':<14}")
    for k, v in out.items():
        ld = v["label_dist"]
        lab = f"{ld['none']}/{ld['somewhat']}/{ld['strongly']}"
        ap = f"{v['probe_auc_strongly_vs_rest']:.3f}" if v['probe_auc_strongly_vs_rest'] is not None else "n/a"
        at = f"{v['tfidf_auc_strongly_vs_rest']:.3f}" if v['tfidf_auc_strongly_vs_rest'] is not None else "n/a"
        print(f"{k:<24} {v['n']:<4} {lab:<18} {v['probe_acc']:<10.3f} {v['tfidf_acc']:<10.3f} {ap:<14} {at:<14}")

    out_path = Path("auth_projection/data/v1_lexical_twin_eval.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
