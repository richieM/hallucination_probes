"""B4: full-position steering on REAL held-out conversations.

The clean version of full-position steering. Take real labeled conversations from
the held-out test split, cut the prefix through the last user turn, generate the
assistant's response with α·v injected at every token during generation. Sweep α
across multiple values; locked sampling seed across all alphas so only steering
strength changes.

Vector v = mean(strongly user residuals) − mean(none user residuals) at the
chosen layer (computed from training-split activations, never test-split).

Run:
    uv run python -m auth_projection.steer_real_conversations --layer 14
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from dotenv import load_dotenv

from utils.file_utils import load_jsonl
from utils.model_utils import get_device, get_model_layers, load_model_and_tokenizer

from .train_probe import split_by_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


def compute_steering_vector(records: List[Dict], layer: int,
                            vector_source: str = "last_token_act") -> torch.Tensor:
    field = vector_source
    src = [r[field][layer].float() for r in records
           if r["label"] == "none" and field in r]
    tgt = [r[field][layer].float() for r in records
           if r["label"] == "strongly" and field in r]
    if not src or not tgt:
        raise ValueError(f"Empty class for {field}: |none|={len(src)} |strongly|={len(tgt)}")
    return torch.stack(tgt).mean(0) - torch.stack(src).mean(0)


def make_steering_hook(steer_vec: torch.Tensor, alpha: float):
    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            h = h + alpha * steer_vec.to(h.dtype).to(h.device)
            return (h,) + output[1:]
        return output + alpha * steer_vec.to(output.dtype).to(output.device)
    return hook


@torch.no_grad()
def generate_with_steering(
    model, tokenizer, device, conv_messages: List[Dict],
    layer_idx_zero: int, steer_vec: torch.Tensor, alpha: float,
    max_new_tokens: int = 350, temperature: float = 0.7, seed: int = 0,
) -> str:
    """Build the chat-template prefix for the conv messages (which should end on a
    user turn), then generate the assistant response with α·v injected at every
    token during generation."""
    full = tokenizer.apply_chat_template(conv_messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(full, return_tensors="pt").to(device)

    layer = get_model_layers(model)[layer_idx_zero]
    handle = layer.register_forward_hook(make_steering_hook(steer_vec, alpha))
    torch.manual_seed(seed)
    try:
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    finally:
        handle.remove()

    gen = out[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def cut_through_last_user_turn(conversation: List[Dict]) -> Optional[List[Dict]]:
    """Return the conversation prefix through the LAST user turn (inclusive).
    The next thing to be generated is the assistant's reply to that user turn."""
    last_user_idx = None
    for i, t in enumerate(conversation):
        if t["role"] == "user":
            last_user_idx = i
    if last_user_idx is None:
        return None
    return [{"role": t["role"], "content": t["content"]}
            for t in conversation[: last_user_idx + 1]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", type=Path,
                        default=Path("auth_projection/data/v3_activations.pt"))
    parser.add_argument("--test_source", type=Path,
                        default=Path("auth_projection/data/v1_labeled.jsonl"))
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--alphas", default="-4,-2,-1,0,1,2,4")
    parser.add_argument("--max_new_tokens", type=int, default=350)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_test_convs", type=int, default=39,
                        help="Number of held-out test convs to steer (default: all 39).")
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output_path", type=Path,
                        default=Path("auth_projection/data/v3b_steering_real_conversations.json"))
    parser.add_argument("--vector_source", default="last_token_act",
                        choices=["last_token_act", "assistant_start_token_act"],
                        help="Which saved activation field to derive the steering vector from.")
    args = parser.parse_args()

    # Compute steering vector from training split (never look at test seeds)
    records = torch.load(args.activations, map_location="cpu")
    train_recs, _ = split_by_seed(records, args.test_frac, args.split_seed)
    v = compute_steering_vector(train_recs, args.layer, vector_source=args.vector_source)
    logger.info(f"Steering vector: layer={args.layer} | source={args.vector_source} "
                f"| ||v||={v.norm().item():.3f} | hidden={v.shape[-1]}")

    if args.layer == 0:
        raise ValueError("Cannot hook before embedding output; pick layer >= 1.")
    layer_idx_zero = args.layer - 1

    # Identify held-out test conversations
    test_seed_ids = {r["seed_id"] for r in records} - {r["seed_id"] for r in train_recs}
    convs = load_jsonl(args.test_source)
    test_convs = [c for c in convs if c["seed_id"] in test_seed_ids]
    test_convs = test_convs[: args.n_test_convs]
    logger.info(f"Held-out test conversations to steer: {len(test_convs)}")

    logger.info(f"Loading {args.model_name}...")
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    model.eval()
    device = get_device()

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    out: List[Dict] = []
    for ci, conv in enumerate(test_convs):
        prefix = cut_through_last_user_turn(conv["conversation"])
        if prefix is None:
            logger.warning(f"Skipping {conv['seed_id'][:8]}: no user turn found")
            continue
        # The label of the LAST user turn (so we can group by ground-truth tier later)
        last_user_index = sum(1 for m in prefix if m["role"] == "user") - 1
        last_user_label = next(
            (tl["label"] for tl in (conv.get("turn_labels") or [])
             if tl["turn_index"] == last_user_index), None
        )

        for alpha in alphas:
            text = generate_with_steering(
                model, tokenizer, device,
                conv_messages=prefix,
                layer_idx_zero=layer_idx_zero, steer_vec=v, alpha=alpha,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, seed=args.seed,
            )
            out.append({
                "seed_id": conv["seed_id"],
                "topic": conv.get("topic"),
                "target_tier": conv.get("target_tier"),
                "lexical_twin_kind": conv.get("lexical_twin_kind"),
                "last_user_index": last_user_index,
                "last_user_label": last_user_label,
                "last_user_text": prefix[-1]["content"],
                "n_user_turns_in_prefix": last_user_index + 1,
                "alpha": alpha,
                "completion": text,
            })
            logger.info(f"[{ci+1}/{len(test_convs)} {conv['seed_id'][:8]}] "
                        f"α={alpha:+.1f} ({last_user_label}) | {text[:120]!r}")
        # Save partial progress after each conversation
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(out, f, indent=2)

    logger.info(f"Saved {len(out)} steering samples -> {args.output_path}")


if __name__ == "__main__":
    main()
