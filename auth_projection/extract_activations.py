"""Extract Llama residual-stream activations at user-token positions.

Run:
    uv run python -m auth_projection.extract_activations
    uv run python -m auth_projection.extract_activations --dense_layers 12,16,20,24
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from dotenv import load_dotenv
from tqdm import tqdm

from utils.file_utils import load_jsonl
from utils.model_utils import get_device, get_num_layers, load_model_and_tokenizer
from utils.tokenization import find_string_in_tokens

from .data_models import Conversation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


DEFAULT_INPUT = Path(__file__).parent / "data" / "labeled.jsonl"
DEFAULT_OUTPUT = Path(__file__).parent / "data" / "activations.pt"


def find_turn_slices(messages, input_ids, tokenizer):
    """For each chat-template message, find the slice of token positions containing its content.
    Advances a cursor so later turns search only in remaining tokens (avoids matching repeats earlier).
    """
    slices = []
    cursor = 0
    for msg in messages:
        content = msg["content"]
        try:
            sub = input_ids[cursor:]
            s = find_string_in_tokens(content, sub, tokenizer)
            global_slc = slice(s.start + cursor, s.stop + cursor)
            slices.append(global_slc)
            cursor = global_slc.stop
        except (AssertionError, ValueError) as e:
            logger.warning(
                f"Could not locate {msg['role']}-turn content in tokens "
                f"({content[:60]!r}...): {e}"
            )
            slices.append(None)
    return slices


def extract_for_conversation(
    conv: Conversation,
    model,
    tokenizer,
    device: torch.device,
    dense_layers: Optional[List[int]] = None,
) -> Optional[List[Dict]]:
    if conv.turn_labels is None:
        return None

    messages = [{"role": t.role, "content": t.content} for t in conv.conversation]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    input_ids = inputs.input_ids[0]

    turn_slices = find_turn_slices(messages, input_ids, tokenizer)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    # tuple of (n_layers + 1) tensors of shape [1, seq, hidden]
    hidden_states = torch.stack(outputs.hidden_states, dim=0).squeeze(1)

    labels_by_idx = {tl.turn_index: tl for tl in conv.turn_labels}
    records = []
    user_idx = 0
    for slc, msg in zip(turn_slices, messages):
        if msg["role"] != "user":
            continue
        tl = labels_by_idx.get(user_idx)
        if tl is None or slc is None:
            user_idx += 1
            continue

        last_pos = slc.stop - 1
        last_token_act = hidden_states[:, last_pos, :].cpu().to(torch.bfloat16).clone()

        record = {
            "seed_id": conv.seed_id,
            "turn_index": user_idx,
            "label": tl.label,
            "topic": conv.topic,
            "target_tier": conv.target_tier,
            "arc_shape": conv.arc_shape,
            "lexical_twin_kind": conv.lexical_twin_kind,
            "uses_lexical_twin": conv.uses_lexical_twin,
            "n_user_tokens_in_turn": slc.stop - slc.start,
            "last_token_act": last_token_act,  # [n_layers+1, hidden]
        }
        if dense_layers:
            dense = hidden_states[dense_layers, slc, :].cpu().to(torch.bfloat16).clone()
            # shape: [len(dense_layers), n_tokens_in_turn, hidden]
            record["dense_act"] = dense
            record["dense_layers"] = list(dense_layers)
        records.append(record)
        user_idx += 1
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--dense_layers",
        default="",
        help="Comma-separated layer indices (in 0..n_layers) to ALSO save dense per-token "
        "activations for. Empty = sparse only (last user-token at all layers). For Probe A.",
    )
    parser.add_argument("--max_convs", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=50)
    args = parser.parse_args()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    dense_layers = [int(x) for x in args.dense_layers.split(",") if x.strip()] or None

    logger.info(f"Loading {args.model_name}...")
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    model.eval()
    device = get_device()
    logger.info(f"Model loaded on {device} | num_layers={get_num_layers(args.model_name)}")

    rows = load_jsonl(args.input_path)
    if args.max_convs:
        rows = rows[: args.max_convs]
    convs = [Conversation.model_validate(r) for r in rows]

    existing_records: List[Dict] = []
    existing_keys = set()
    if args.output_path.exists():
        existing_records = torch.load(args.output_path, map_location="cpu")
        existing_keys = {(r["seed_id"], r["turn_index"]) for r in existing_records}
        logger.info(f"Resuming: {len(existing_keys)} (seed, turn) pairs already extracted")

    all_records = list(existing_records)
    new_count = 0
    for conv in tqdm(convs, desc="Extracting"):
        n_user = sum(1 for t in conv.conversation if t.role == "user")
        already_for_seed = sum(1 for k in existing_keys if k[0] == conv.seed_id)
        if already_for_seed >= n_user:
            continue

        recs = extract_for_conversation(
            conv, model, tokenizer, device, dense_layers=dense_layers
        )
        if recs is None:
            continue
        for r in recs:
            key = (r["seed_id"], r["turn_index"])
            if key in existing_keys:
                continue
            all_records.append(r)
            existing_keys.add(key)
            new_count += 1
        if new_count > 0 and new_count % args.save_every == 0:
            torch.save(all_records, args.output_path)
            logger.info(f"Saved checkpoint at {len(all_records)} records")

    torch.save(all_records, args.output_path)
    logger.info(f"Saved {len(all_records)} turn-level activations -> {args.output_path}")


if __name__ == "__main__":
    main()
