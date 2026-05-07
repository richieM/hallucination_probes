"""User-position-only steering: apply the authority-projection vector ONLY at
user-turn token positions in the conversation prefix, then generate the assistant's
response with NO perturbation on the generated tokens.

Tests whether the steering vector represents *user state on the model's side* (which
should propagate through KV cache to influence assistant generation) vs just being a
"deferential language" direction (which would only affect output if applied during
generation, as in steer.py).

Run:
    uv run python -m auth_projection.steer_user_positions --layer 17
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from dotenv import load_dotenv

from utils.file_utils import load_jsonl
from utils.model_utils import get_device, get_model_layers, load_model_and_tokenizer
from utils.tokenization import find_string_in_tokens

from .train_probe import split_by_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


DEFAULT_ACTIVATIONS = Path(__file__).parent / "data" / "v1_activations.pt"
DEFAULT_TEST_SOURCE = Path(__file__).parent / "data" / "v1_labeled.jsonl"
DEFAULT_OUTPUT = Path(__file__).parent / "data" / "v1_steering_user_positions.json"


def compute_steering_vector(records: List[Dict], layer: int) -> torch.Tensor:
    src = [r["last_token_act"][layer].float() for r in records if r["label"] == "none"]
    tgt = [r["last_token_act"][layer].float() for r in records if r["label"] == "strongly"]
    return torch.stack(tgt).mean(0) - torch.stack(src).mean(0)


def find_turn_slices(messages, input_ids, tokenizer):
    slices = []
    cursor = 0
    for msg in messages:
        try:
            sub = input_ids[cursor:]
            s = find_string_in_tokens(msg["content"], sub, tokenizer)
            global_slc = slice(s.start + cursor, s.stop + cursor)
            slices.append((msg["role"], global_slc))
            cursor = global_slc.stop
        except (AssertionError, ValueError):
            slices.append((msg["role"], None))
    return slices


def build_user_position_mask(messages, input_ids, tokenizer) -> torch.Tensor:
    """1.0 at positions inside any user-turn content, 0.0 elsewhere."""
    mask = torch.zeros(input_ids.shape[0], dtype=torch.float32)
    for role, slc in find_turn_slices(messages, input_ids, tokenizer):
        if role == "user" and slc is not None:
            mask[slc] = 1.0
    return mask


class UserPositionSteeringHook:
    """Applies steer_vec * alpha at masked positions on the FIRST forward pass only
    (the prefix encoding). Subsequent forwards (per-token generation) are untouched."""

    def __init__(self, steer_vec: torch.Tensor, alpha: float, user_position_mask: torch.Tensor):
        self.steer_vec = steer_vec
        self.alpha = alpha
        self.mask = user_position_mask  # [prefix_len]
        self.applied = False

    def __call__(self, module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = None
        seq_len = h.shape[1]
        if seq_len == self.mask.shape[0] and not self.applied:
            mask = self.mask.to(h.device).to(h.dtype)
            v = self.steer_vec.to(h.dtype).to(h.device)
            h = h + self.alpha * mask[None, :, None] * v[None, None, :]
            self.applied = True
        # Else: post-prefix forward (1 token at a time during generation), no-op.
        if rest is not None:
            return (h,) + rest
        return h


@torch.no_grad()
def generate_with_user_position_steering(
    model, tokenizer, device,
    conversation: List[Dict], user_turn_index: int,
    layer_idx_zero: int, steer_vec: torch.Tensor, alpha: float,
    max_new_tokens: int = 250, temperature: float = 0.7, seed: int = 0,
) -> Tuple[str, int]:
    """Build prefix through user_turn_index, mask user positions, steer, generate."""
    user_count = 0
    cut = None
    for i, t in enumerate(conversation):
        if t["role"] == "user":
            if user_count == user_turn_index:
                cut = i + 1
                break
            user_count += 1
    if cut is None:
        raise ValueError(f"user_turn_index={user_turn_index} not in conv")
    prefix = [{"role": t["role"], "content": t["content"]} for t in conversation[:cut]]

    full = tokenizer.apply_chat_template(prefix, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(full, return_tensors="pt").to(device)
    input_ids = inputs.input_ids[0]

    user_mask = build_user_position_mask(prefix, input_ids, tokenizer)
    n_user_tokens = int(user_mask.sum().item())

    layer = get_model_layers(model)[layer_idx_zero]
    hook = UserPositionSteeringHook(steer_vec, alpha, user_mask)
    handle = layer.register_forward_hook(hook)

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
    return tokenizer.decode(gen, skip_special_tokens=True), n_user_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", type=Path, default=DEFAULT_ACTIVATIONS)
    parser.add_argument("--test_source", type=Path, default=DEFAULT_TEST_SOURCE)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--alphas", default="-8,-4,-2,0,2,4,8")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_test_convs", type=int, default=8,
                        help="Number of held-out test conversations to steer on.")
    parser.add_argument("--user_turn_index", type=int, default=2,
                        help="Steer at this user-turn (must be present in chosen convs).")
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    records = torch.load(args.activations, map_location="cpu")
    train_recs, _ = split_by_seed(records, args.test_frac, args.split_seed)
    v = compute_steering_vector(train_recs, args.layer)
    logger.info(f"Vector: layer={args.layer} | norm={v.norm().item():.3f}")

    layer_idx_zero = args.layer - 1

    # Pick test conversations from held-out seeds with at least user_turn_index+1 user turns
    convs = load_jsonl(args.test_source)
    test_seed_ids = {r["seed_id"] for r in records} - {r["seed_id"] for r in train_recs}
    candidates = []
    for c in convs:
        if c["seed_id"] not in test_seed_ids:
            continue
        n_user = sum(1 for t in c["conversation"] if t["role"] == "user")
        if n_user > args.user_turn_index:
            candidates.append(c)
    candidates = candidates[: args.n_test_convs]
    logger.info(f"Test conversations selected: {len(candidates)}")

    logger.info(f"Loading {args.model_name}...")
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    model.eval()
    device = get_device()

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    out: List[Dict] = []
    for c in candidates:
        for alpha in alphas:
            text, n_u = generate_with_user_position_steering(
                model, tokenizer, device,
                c["conversation"], args.user_turn_index,
                layer_idx_zero=layer_idx_zero, steer_vec=v, alpha=alpha,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, seed=args.seed,
            )
            target_user_text = c["conversation"][2 * args.user_turn_index]["content"][:160]
            out.append({
                "seed_id": c["seed_id"],
                "topic": c.get("topic"),
                "target_user_turn_index": args.user_turn_index,
                "target_user_turn_text": target_user_text,
                "alpha": alpha,
                "n_user_tokens_perturbed": n_u,
                "completion": text,
            })
            logger.info(f"[{c['seed_id'][:8]}] alpha={alpha:+.1f} | n_user_tok={n_u} | "
                        f"{text[:130]!r}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved {len(out)} samples -> {args.output_path}")


if __name__ == "__main__":
    main()
