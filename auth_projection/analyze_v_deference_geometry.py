"""Free analyses of the deference vector / user-state activation geometry, run locally
on existing v6c activations.

Produces:
- PCA: how much of strongly-vs-none separation lives on PC1?
- Cross-layer cosine similarity of v_deference at every layer (is it the same feature evolving, or different per layer?)
- Topic correlation: does v_deference vary by conversation topic?
- Top-k PC reconstruction: how well does steering with just PC1 reproduce the full v_def steering effect?

Output: auth_projection/data/v6c_geometry_analysis.json + a textual summary printed to stdout.

Run:
    uv run python -m auth_projection.analyze_v_deference_geometry
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from auth_projection.train_probe import split_by_seed
from auth_projection.steer_real_conversations import compute_steering_vector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    DATA = Path("auth_projection/data")
    records = torch.load(DATA / "v6c_activations.pt", map_location="cpu")
    train_recs, _ = split_by_seed(records, 0.2, 42)
    LAYER = 14
    N_LAYERS = records[0]["last_token_act"].shape[0]
    HIDDEN = records[0]["last_token_act"].shape[1]
    print(f"Loaded {len(records)} records | n_layers+1={N_LAYERS} | hidden={HIDDEN}")

    out = {}

    # ============ 1. PCA on L14 strongly vs none ============
    print(f"\n=== 1. PCA on L{LAYER} strongly vs none (training split, n_train={len(train_recs)}) ===")
    strongly = np.stack([r["last_token_act"][LAYER].float().numpy() for r in train_recs if r["label"] == "strongly"])
    none_ = np.stack([r["last_token_act"][LAYER].float().numpy() for r in train_recs if r["label"] == "none"])
    print(f"strongly: {strongly.shape}, none: {none_.shape}")
    X = np.concatenate([strongly, none_], axis=0)
    y = np.concatenate([np.ones(strongly.shape[0]), np.zeros(none_.shape[0])])
    X_centered = X - X.mean(0)
    # PCA via SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    explained_var = S ** 2 / np.sum(S ** 2)
    print(f"Explained variance: PC1={explained_var[0]:.3f}, PC2={explained_var[1]:.3f}, PC3={explained_var[2]:.3f}, PC4={explained_var[3]:.3f}, PC5={explained_var[4]:.3f}")

    # How well does PC1 separate the classes?
    pc_proj = X_centered @ Vt.T  # [n, n_pc]
    for k in [1, 2, 3, 5, 10]:
        # Use simple LDA-style: project onto first k PCs, train logistic regression
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        Xk = pc_proj[:, :k]
        clf = LogisticRegression(max_iter=2000).fit(Xk, y)
        acc = accuracy_score(y, clf.predict(Xk))
        print(f"  Top-{k} PCs strongly-vs-none train acc: {acc:.3f}")

    # Cosine similarity of v_def with PC1
    v_def = compute_steering_vector(train_recs, LAYER, vector_source="last_token_act").numpy()
    pc1 = Vt[0]
    cos_v_pc1 = float(np.dot(v_def, pc1) / (np.linalg.norm(v_def) * np.linalg.norm(pc1)))
    cos_v_pcs = [float(np.dot(v_def, Vt[k]) / (np.linalg.norm(v_def) * np.linalg.norm(Vt[k]))) for k in range(10)]
    print(f"Cosine(v_def, PC1) = {cos_v_pc1:.3f}")
    print(f"Cosines v_def with first 10 PCs: {[f'{c:+.3f}' for c in cos_v_pcs]}")

    # ‖projection of v_def onto top-k PC subspace‖ / ‖v_def‖
    for k in [1, 2, 3, 5, 10]:
        Pk = Vt[:k].T @ Vt[:k]  # projection onto top-k PCs
        v_proj = Pk @ v_def
        ratio = float(np.linalg.norm(v_proj) / np.linalg.norm(v_def))
        print(f"  ‖proj(v_def, top-{k} PCs)‖ / ‖v_def‖ = {ratio:.3f}")

    out["pca"] = {
        "explained_var_top10": [float(x) for x in explained_var[:10]],
        "cos_v_def_PC1": cos_v_pc1,
        "cos_v_def_top10_PCs": cos_v_pcs,
        "n_strongly_train": int(strongly.shape[0]),
        "n_none_train": int(none_.shape[0]),
    }

    # ============ 2. Cross-layer cosine similarity ============
    print(f"\n=== 2. Cross-layer cosine: how does v_deference evolve through the residual stream? ===")
    v_per_layer = []
    for L in range(N_LAYERS):
        try:
            v = compute_steering_vector(train_recs, L, vector_source="last_token_act")
            v_per_layer.append(v.numpy())
        except Exception:
            v_per_layer.append(None)
    norms = [float(np.linalg.norm(v)) if v is not None else None for v in v_per_layer]
    print(f"‖v_def‖ at each layer (every 4th):")
    for L in range(0, N_LAYERS, 4):
        print(f"  L{L:2d}: ‖v‖={norms[L]:.3f}" if norms[L] is not None else f"  L{L:2d}: -")
    # Cosine similarity matrix (just print a few key cells)
    cos_matrix = np.zeros((N_LAYERS, N_LAYERS))
    for i in range(N_LAYERS):
        for j in range(N_LAYERS):
            if v_per_layer[i] is not None and v_per_layer[j] is not None:
                cos_matrix[i, j] = float(np.dot(v_per_layer[i], v_per_layer[j]) / (np.linalg.norm(v_per_layer[i]) * np.linalg.norm(v_per_layer[j]) + 1e-10))
    print(f"\nCross-layer cosine (selected layers vs L14):")
    for L in [0, 5, 10, 13, 14, 15, 17, 21, 25, 30, 32]:
        print(f"  cos(v_L{L:2d}, v_L14) = {cos_matrix[L, 14]:+.3f}")
    out["cross_layer_cosine"] = {
        "cos_with_L14": [float(cos_matrix[L, 14]) for L in range(N_LAYERS)],
        "norms": norms,
    }

    # ============ 3. Topic correlation ============
    print(f"\n=== 3. Topic dependency of v_deference ===")
    # For each topic, build v_def using only that topic's records, compare to global v_def
    by_topic = defaultdict(list)
    for r in train_recs:
        by_topic[r.get("topic", "?")].append(r)
    print(f"Topics: {sorted(by_topic.keys())}")
    out["topic_v_def_cosines"] = {}
    for topic, topic_recs in by_topic.items():
        try:
            v_topic = compute_steering_vector(topic_recs, LAYER, vector_source="last_token_act").numpy()
            cos = float(np.dot(v_topic, v_def) / (np.linalg.norm(v_topic) * np.linalg.norm(v_def)))
            print(f"  topic={topic:<25} n={len(topic_recs):3d}  ||v||={np.linalg.norm(v_topic):.3f}  cos(v_topic, v_global)={cos:+.3f}")
            out["topic_v_def_cosines"][topic] = {"n": len(topic_recs), "norm": float(np.linalg.norm(v_topic)), "cos_with_global": cos}
        except ValueError as e:
            print(f"  topic={topic}: {e}")

    # ============ 4. Twin-failure case analysis ============
    print(f"\n=== 4. submission_voice probe failures (qualitative) ===")
    # Read v1_lexical_twin_eval.json — get the submission_voice cases
    twin_eval = json.load(open(DATA / "v1_lexical_twin_eval.json"))
    # Look at structure
    print(f"Keys: {list(twin_eval.keys())}")
    if "submission_voice" in twin_eval:
        sv = twin_eval["submission_voice"]
        print(f"submission_voice cell: n={sv.get('n')}, probe_acc={sv.get('probe_acc')}, tfidf_acc={sv.get('tfidf_acc')}")
    out["twin_eval_summary"] = {k: v for k, v in twin_eval.items() if isinstance(v, dict) and 'n' in v}

    # ============ 5. Save ============
    output_path = DATA / "v6c_geometry_analysis.json"
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {output_path}")


if __name__ == "__main__":
    main()
