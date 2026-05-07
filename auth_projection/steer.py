"""Compute an authority-projection steering vector at the probe layer and sweep alpha
during generation to test whether the direction is causally meaningful.

Vector definition:
    v = mean(last_token_act[layer] | label="strongly")
      - mean(last_token_act[layer] | label="none")
    (computed on the train split; test seeds held out.)

Steering is implemented as a forward hook on `model.model.layers[layer-1]` that
adds alpha * v to its output's hidden_states (broadcast across all token positions).

Run:
    uv run python -m auth_projection.steer --layer 17
    uv run python -m auth_projection.steer --layer 17 --alphas -16,-8,-4,-2,0,2,4,8,16
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from dotenv import load_dotenv

from utils.model_utils import get_device, get_model_layers, load_model_and_tokenizer

from .train_probe import LABEL_TO_INT, split_by_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


DEFAULT_ACTIVATIONS = Path(__file__).parent / "data" / "v1_activations.pt"
DEFAULT_OUTPUT = Path(__file__).parent / "data" / "v1_steering_samples.json"


# A small set of seed prefixes — we generate a continuation as if the user is
# expressing themselves under steering. Probe was trained on user turns, so we
# nudge the residual at the same direction it learned to detect.
SEED_PROMPTS = [
    {
        "name": "career_pivot",
        "system": "You are roleplaying as a HUMAN USER chatting to an AI assistant. "
                  "Continue speaking only as the user — express your situation, thoughts, "
                  "and what you want from the assistant. Do not impersonate the assistant.",
        "user_seed": "I've been thinking about leaving my job to start something of my own. "
                     "Here's where I am with it: ",
    },
    {
        "name": "relationship_doubt",
        "system": "You are roleplaying as a HUMAN USER chatting to an AI assistant. "
                  "Continue speaking only as the user — express your situation, thoughts, "
                  "and what you want from the assistant. Do not impersonate the assistant.",
        "user_seed": "My partner and I had a fight last night about money. The thing is, ",
    },
    {
        "name": "creative_block",
        "system": "You are roleplaying as a HUMAN USER chatting to an AI assistant. "
                  "Continue speaking only as the user — express your situation, thoughts, "
                  "and what you want from the assistant. Do not impersonate the assistant.",
        "user_seed": "I'm a writer and I've been stuck on this novel for months. ",
    },
]


def compute_steering_vector(
    records: List[Dict], layer: int, source_label: str = "none", target_label: str = "strongly"
) -> torch.Tensor:
    src = [r["last_token_act"][layer].float() for r in records if r["label"] == source_label]
    tgt = [r["last_token_act"][layer].float() for r in records if r["label"] == target_label]
    if not src or not tgt:
        raise ValueError(f"Empty class: |{source_label}|={len(src)} |{target_label}|={len(tgt)}")
    v = torch.stack(tgt).mean(0) - torch.stack(src).mean(0)
    return v


def make_steering_hook(steer_vec: torch.Tensor, alpha: float):
    """Forward hook on a Qwen/Llama decoder layer. Layer outputs are tuples
    (hidden_states, *rest). We add alpha * steer_vec to hidden_states."""
    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            h = h + alpha * steer_vec.to(h.dtype).to(h.device)
            return (h,) + output[1:]
        else:
            return output + alpha * steer_vec.to(output.dtype).to(output.device)
    return hook


@torch.no_grad()
def generate_with_steering(
    model, tokenizer, device, system: str, user_seed: str,
    layer_idx_zero: int, steer_vec: torch.Tensor, alpha: float,
    max_new_tokens: int = 200, temperature: float = 0.7, seed: int = 0,
) -> str:
    """Build a chat-template prompt that ends mid-user-turn and continue under steering.

    layer_idx_zero is the 0-indexed transformer block whose OUTPUT we hook
    (matches hidden_states[layer_idx_zero + 1] = probe layer).
    """
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_seed}]
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    return tokenizer.decode(gen, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", type=Path, default=DEFAULT_ACTIVATIONS)
    parser.add_argument("--layer", type=int, required=True,
                        help="Probe layer index (matches hidden_states[layer]). "
                        "Steering hook will be on transformer block (layer-1).")
    parser.add_argument("--alphas", default="-16,-8,-4,-2,0,2,4,8,16")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--normalize", action="store_true",
                        help="L2-normalize the steering vector before scaling.")
    args = parser.parse_args()

    records = torch.load(args.activations, map_location="cpu")
    train_recs, _ = split_by_seed(records, args.test_frac, args.split_seed)
    logger.info(f"Loaded {len(records)} records | train: {len(train_recs)}")

    v = compute_steering_vector(train_recs, args.layer, "none", "strongly")
    raw_norm = v.norm().item()
    if args.normalize:
        v = v / max(raw_norm, 1e-8)
    logger.info(f"Steering vector: layer={args.layer} | raw_norm={raw_norm:.3f} | "
                f"used_norm={v.norm().item():.3f} | hidden={v.shape[-1]}")

    if args.layer == 0:
        raise ValueError("Cannot hook before the embedding output; pick layer >= 1.")
    layer_idx_zero = args.layer - 1

    logger.info(f"Loading {args.model_name}...")
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    model.eval()
    device = get_device()

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    out: List[Dict] = []
    for prompt in SEED_PROMPTS:
        for alpha in alphas:
            text = generate_with_steering(
                model, tokenizer, device,
                system=prompt["system"], user_seed=prompt["user_seed"],
                layer_idx_zero=layer_idx_zero, steer_vec=v, alpha=alpha,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, seed=args.seed,
            )
            out.append({
                "prompt_name": prompt["name"],
                "user_seed": prompt["user_seed"],
                "alpha": alpha,
                "completion": text,
                "layer": args.layer,
                "raw_vec_norm": raw_norm,
                "normalized": args.normalize,
            })
            logger.info(f"[{prompt['name']}] alpha={alpha:+.1f} | {text[:140]!r}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved {len(out)} steering samples -> {args.output_path}")


if __name__ == "__main__":
    main()
