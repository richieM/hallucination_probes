"""Read the actual user turns from submission_voice cases and inspect what the
probe gets wrong vs TF-IDF.

submission_voice = user uses deferential VOCABULARY but doesn't actually defer
behaviorally (target tier = 'none'). The probe loses to TF-IDF here at 8B
(0.60 vs 0.80, n=15).

Question: are the failures concentrated in a specific kind of submission-voice
phrasing? Are they about specific words?
"""
import json
import numpy as np
import torch
from collections import Counter
from pathlib import Path

from auth_projection.train_probe import split_by_seed
from utils.file_utils import load_jsonl


def main():
    DATA = Path("auth_projection/data")
    convs = {c["seed_id"]: c for c in load_jsonl(DATA / "v1_labeled.jsonl")}
    records = torch.load(DATA / "v6c_activations.pt", map_location="cpu")
    train_recs, test_recs = split_by_seed(records, 0.2, 42)

    # submission_voice = lexical_twin_kind == 'submission_voice' (target was 'none' but labels can vary)
    sv_test = [r for r in test_recs if r.get("lexical_twin_kind") == "submission_voice"]
    print(f"submission_voice test cases: {len(sv_test)}")
    LABEL_TO_INT = {"none": 0, "somewhat": 1, "strongly": 2}
    INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}

    # Load the probe at L14
    import joblib
    LAYER = 14
    probe = joblib.load(DATA / "v6c_probes" / f"probe_C_layer{LAYER}.joblib")

    # For each, run probe and TF-IDF
    # First, fit TF-IDF the same way train_probe does — quick re-fit
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    def get_text(r, conv_lookup):
        sid = r["seed_id"]; ti = r["turn_index"]
        conv = conv_lookup[sid]
        users = [t for t in conv["conversation"] if t["role"] == "user"]
        return users[ti]["content"]

    train_X_text = [get_text(r, convs) for r in train_recs]
    train_y = [r["label"] for r in train_recs]
    vect = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
    Xtr = vect.fit_transform(train_X_text)
    tfidf_clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, train_y)

    print("\n=== Submission_voice test cases (label=none, BUT user uses deferential vocab) ===\n")
    print(f"{'seed':<10} {'probe_pred':<12} {'tfidf_pred':<12} {'probe_correct':<15} {'tfidf_correct':<15} text")
    probe_acc, tfidf_acc = 0, 0
    for r in sv_test:
        x = r["last_token_act"][LAYER].float().numpy().reshape(1, -1)
        probe_pred_int = probe.predict(x)[0]
        probe_pred = INT_TO_LABEL[probe_pred_int]
        text = get_text(r, convs)
        tfidf_pred = tfidf_clf.predict(vect.transform([text]))[0]
        probe_correct = probe_pred == r["label"]
        tfidf_correct = tfidf_pred == r["label"]
        probe_acc += probe_correct
        tfidf_acc += tfidf_correct
        print(f"  {r['seed_id'][:8]}  true={r['label']:<10} probe={probe_pred:<10} tfidf={tfidf_pred:<10} probe_ok={'✓' if probe_correct else '✗'} tfidf_ok={'✓' if tfidf_correct else '✗'}  {text[:60]!r}")

    print(f"\nProbe acc: {probe_acc}/{len(sv_test)} = {100*probe_acc/len(sv_test):.0f}%")
    print(f"TF-IDF acc: {tfidf_acc}/{len(sv_test)} = {100*tfidf_acc/len(sv_test):.0f}%")

    # Print probe-wrong AND tfidf-right cases
    print("\n=== Cases where probe fails but TF-IDF wins (partial word-anchoring failure) ===")
    for r in sv_test:
        x = r["last_token_act"][LAYER].float().numpy().reshape(1, -1)
        probe_pred = INT_TO_LABEL[probe.predict(x)[0]]
        text = get_text(r, convs)
        tfidf_pred = tfidf_clf.predict(vect.transform([text]))[0]
        if probe_pred != r["label"] and tfidf_pred == r["label"]:
            print(f"\n[{r['seed_id'][:8]}] probe={probe_pred}, tfidf={tfidf_pred} (true={r['label']})")
            print(f"  text: {text!r}")


if __name__ == "__main__":
    main()
