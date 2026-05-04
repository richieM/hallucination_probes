"""Spot-check labeled conversations.

Run as a script:
    uv run python -m auth_projection.spot_check
    uv run python -m auth_projection.spot_check --filter-tier strongly --limit 10
    uv run python -m auth_projection.spot_check --only-disagreements

Or open in VS Code / Jupyter — `# %%` markers split into runnable cells.
"""

# %%
import argparse
from collections import Counter
from pathlib import Path
from typing import List, Optional

from utils.file_utils import load_jsonl
from .data_models import Conversation


DEFAULT_LABELED = Path(__file__).parent / "data" / "labeled.jsonl"


# %%
def load_conversations(path: Path = DEFAULT_LABELED) -> List[Conversation]:
    if not path.exists():
        print(f"No file at {path}. Run the labeling pipeline first.")
        return []
    rows = load_jsonl(path)
    return [Conversation.model_validate(r) for r in rows]


# %%
def print_summary(convs: List[Conversation]) -> None:
    print(f"Total conversations: {len(convs)}")
    if not convs:
        return

    target_dist = Counter(c.target_tier for c in convs)
    print(f"\nGenerator target_tier distribution: {dict(target_dist)}")

    label_dist: Counter = Counter()
    for c in convs:
        if c.turn_labels:
            for tl in c.turn_labels:
                label_dist[tl.label] += 1
    print(f"Labeler turn-label distribution: {dict(label_dist)}")

    n_lex_twin = sum(1 for c in convs if c.uses_lexical_twin)
    print(f"Lexical-twin conversations: {n_lex_twin}/{len(convs)}")

    # target vs final-turn label agreement
    agree = 0
    total = 0
    for c in convs:
        if not c.turn_labels:
            continue
        final = c.turn_labels[-1].label
        total += 1
        if final == c.target_tier:
            agree += 1
    if total:
        print(f"\nFinal-turn label vs generator target_tier agreement: {agree}/{total} ({agree/total:.0%})")

    topic_counts = Counter(c.topic for c in convs)
    print(f"\nTopic coverage: {len(topic_counts)} topics, "
          f"min={min(topic_counts.values())}, max={max(topic_counts.values())}")


# %%
def pretty_print(conv: Conversation, show_rationale: bool = True) -> None:
    print("=" * 80)
    print(f"seed_id: {conv.seed_id}")
    print(f"topic: {conv.topic}")
    print(f"target_tier: {conv.target_tier}    "
          f"uses_lexical_twin: {conv.uses_lexical_twin}")
    print(f"persona: {conv.persona.to_prompt_description()}")
    print("-" * 80)

    user_idx = 0
    label_by_idx = {}
    if conv.turn_labels:
        label_by_idx = {tl.turn_index: tl for tl in conv.turn_labels}

    for turn in conv.conversation:
        if turn.role == "user":
            tl = label_by_idx.get(user_idx)
            label_str = f"[{tl.label}]" if tl else "[unlabeled]"
            print(f"\nUSER (turn {user_idx}) {label_str}")
            print(f"  {turn.content}")
            if tl and show_rationale:
                print(f"    rationale: {tl.rationale}")
            user_idx += 1
        else:
            print(f"\nASSISTANT")
            print(f"  {turn.content}")
    print()


# %%
def filter_convs(
    convs: List[Conversation],
    target_tier: Optional[str] = None,
    uses_lexical_twin: Optional[bool] = None,
    only_disagreements: bool = False,
    only_lexical_twins: bool = False,
) -> List[Conversation]:
    result = convs
    if target_tier:
        result = [c for c in result if c.target_tier == target_tier]
    if uses_lexical_twin is not None:
        result = [c for c in result if c.uses_lexical_twin == uses_lexical_twin]
    if only_lexical_twins:
        result = [c for c in result if c.uses_lexical_twin]
    if only_disagreements:
        out = []
        for c in result:
            if not c.turn_labels:
                continue
            final = c.turn_labels[-1].label
            if final != c.target_tier:
                out.append(c)
        result = out
    return result


# %%
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_LABELED)
    parser.add_argument("--filter-tier", choices=["none", "somewhat", "strongly"], default=None)
    parser.add_argument("--only-lexical-twins", action="store_true")
    parser.add_argument("--only-disagreements", action="store_true",
                        help="Only show convs where final turn label != target_tier")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--no-rationale", action="store_true")
    args = parser.parse_args()

    convs = load_conversations(args.input)
    print_summary(convs)
    print()

    filtered = filter_convs(
        convs,
        target_tier=args.filter_tier,
        only_lexical_twins=args.only_lexical_twins,
        only_disagreements=args.only_disagreements,
    )
    print(f"Showing {min(len(filtered), args.limit)} of {len(filtered)} matching conversations.\n")

    for c in filtered[:args.limit]:
        pretty_print(c, show_rationale=not args.no_rationale)


if __name__ == "__main__":
    main()
