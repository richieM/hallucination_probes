"""B2: forward-pass the hand-built test conversations through Llama 8B,
extract activations, and run the trained probe at L14 to compare predicted
tier against the expected (hand-assigned) tier for each conversation.

Saves: per-conversation prediction, predicted tier, expected tier, agreement,
plus probabilities for each class.

Run:
    uv run python -m auth_projection.run_handbuilt_testbed --layer 14 \\
        --probes_dir auth_projection/data/v3b_probes
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import torch

from utils.file_utils import load_jsonl
from utils.model_utils import get_device, get_num_layers, load_model_and_tokenizer
from utils.tokenization import find_string_in_tokens

from .train_probe import LABEL_TO_INT, INT_TO_LABEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_turn_slices(messages, input_ids, tokenizer):
    slices = []
    cursor = 0
    for msg in messages:
        try:
            sub = input_ids[cursor:]
            s = find_string_in_tokens(msg["content"], sub, tokenizer)
            global_slc = slice(s.start + cursor, s.stop + cursor)
            slices.append(global_slc)
            cursor = global_slc.stop
        except (AssertionError, ValueError):
            slices.append(None)
    return slices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testbed_path", type=Path,
                        default=Path("auth_projection/data/handbuilt_testbed.jsonl"))
    parser.add_argument("--probes_dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output_path", type=Path,
                        default=Path("auth_projection/data/v3b_handbuilt_predictions.json"))
    args = parser.parse_args()

    testbed = load_jsonl(args.testbed_path)
    logger.info(f"Loaded {len(testbed)} hand-built test conversations")

    # Load probe
    probe_path = args.probes_dir / f"probe_C_layer{args.layer}.joblib"
    clf = joblib.load(probe_path)
    logger.info(f"Loaded probe: {probe_path}")

    logger.info(f"Loading {args.model_name}...")
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    model.eval()
    device = get_device()

    results = []
    correct = 0
    for conv in testbed:
        messages = [{"role": t["role"], "content": t["content"]} for t in conv["conversation"]]
        full = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = tokenizer(full, return_tensors="pt").to(device)
        input_ids = inputs.input_ids[0]
        turn_slices = find_turn_slices(messages, input_ids, tokenizer)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        hidden_states = torch.stack(outputs.hidden_states, dim=0).squeeze(1)

        # Find the LAST user-turn slice (testbed conv ends mid-turn or with a final user turn)
        last_user_slc = None
        for slc, msg in zip(turn_slices, messages):
            if msg["role"] == "user" and slc is not None:
                last_user_slc = slc
        if last_user_slc is None:
            logger.warning(f"Skipping {conv['seed_id']}: no user turn located")
            continue

        last_pos = last_user_slc.stop - 1
        act_at_layer = hidden_states[args.layer, last_pos, :].float().cpu().numpy()
        probs = clf.predict_proba(act_at_layer[None, :])[0]
        pred_int = int(clf.predict(act_at_layer[None, :])[0])
        pred_label = INT_TO_LABEL[pred_int]
        expected_label = conv["tier"]
        match = pred_label == expected_label
        if match:
            correct += 1

        # Build a probs dict aligned to clf.classes_
        prob_dict = {}
        for cls_int, p in zip(clf.classes_, probs):
            prob_dict[INT_TO_LABEL[int(cls_int)]] = float(p)

        results.append({
            "seed_id": conv["seed_id"],
            "topic": conv["topic"],
            "twin_kind": conv["twin_kind"],
            "expected_tier": expected_label,
            "predicted_tier": pred_label,
            "match": match,
            "probs": prob_dict,
            "rationale": conv.get("rationale", ""),
            "last_user_text": messages[-1]["content"][:200] if messages[-1]["role"] == "user" else "",
        })
        logger.info(f"  {conv['seed_id'][:25]:<25} | expected={expected_label:<9} pred={pred_label:<9} "
                    f"{'✓' if match else '✗'} | probs={prob_dict}")

    overall = {
        "n": len(results),
        "n_correct": correct,
        "accuracy": correct / len(results) if results else 0.0,
        "by_twin_kind": {},
        "results": results,
    }
    from collections import defaultdict
    bucket = defaultdict(list)
    for r in results:
        bucket[r["twin_kind"]].append(r)
    for k, items in bucket.items():
        n_correct_k = sum(1 for r in items if r["match"])
        overall["by_twin_kind"][k] = {
            "n": len(items),
            "n_correct": n_correct_k,
            "accuracy": n_correct_k / len(items) if items else 0.0,
        }

    print(f"\n=== Hand-built testbed at L{args.layer} ===")
    print(f"  Overall: {correct}/{len(results)} ({100*correct/len(results):.0f}%)")
    for k, m in overall["by_twin_kind"].items():
        print(f"  {k:<20} {m['n_correct']}/{m['n']} ({100*m['accuracy']:.0f}%)")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(overall, f, indent=2)
    logger.info(f"Saved -> {args.output_path}")


if __name__ == "__main__":
    main()
