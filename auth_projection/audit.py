"""Interactive human-vs-labeler audit tool.

Sample N random user turns from labeled.jsonl, walk through them one at a time,
record whether you agree with the labeler. Saves progress as you go so you can
quit and resume.

Run:
    uv run python -m auth_projection.audit --n 30
    uv run python -m auth_projection.audit --score-only
    uv run python -m auth_projection.audit --reset --n 30   # start a fresh audit
"""

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

from utils.file_utils import load_jsonl, save_jsonl

DEFAULT_LABELED = Path(__file__).parent / "data" / "labeled.jsonl"
DEFAULT_AUDIT = Path(__file__).parent / "data" / "audit.jsonl"


def sample_turns(labeled_path: Path, n: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    rows = load_jsonl(labeled_path)
    samples = []
    for r in rows:
        if not r.get("turn_labels"):
            continue
        for tl in r["turn_labels"]:
            samples.append(
                {
                    "seed_id": r["seed_id"],
                    "topic": r["topic"],
                    "persona": r["persona"],
                    "target_tier": r["target_tier"],
                    "arc_shape": r.get("arc_shape"),
                    "lexical_twin_kind": r.get("lexical_twin_kind"),
                    "uses_lexical_twin": r.get("uses_lexical_twin"),
                    "conversation": r["conversation"],
                    "turn_index": tl["turn_index"],
                    "labeler_label": tl["label"],
                    "labeler_rationale": tl["rationale"],
                    "human_label": None,
                    "human_note": "",
                }
            )
    rng.shuffle(samples)
    return samples[:n]


def render_sample(sample: Dict) -> str:
    p = sample["persona"]
    persona_str = (
        f"{p['age_bucket']}, {p['communication_style']}, {p['prior_ai_experience']} AI use, "
        f"{p['life_context']}, goal={p['conversational_goal']}"
    )

    lex = sample.get("lexical_twin_kind") or (
        "twin(legacy)" if sample.get("uses_lexical_twin") else "match(legacy)"
        if sample.get("uses_lexical_twin") is not None
        else "?"
    )
    arc = sample.get("arc_shape") or "?"

    lines = []
    lines.append("=" * 80)
    lines.append(f"Topic: {sample['topic']}")
    lines.append(f"Persona: {persona_str}")
    lines.append(
        f"Generator target_tier: {sample['target_tier']}  |  "
        f"arc_shape: {arc}  |  "
        f"lexical_twin_kind: {lex}"
    )
    lines.append("-" * 80)

    target_idx = sample["turn_index"]
    user_idx = 0
    for t in sample["conversation"]:
        role = t["role"]
        content = t["content"]
        if role == "user":
            marker = " ← TARGET TURN" if user_idx == target_idx else ""
            lines.append(f"\n[USER turn {user_idx}]{marker}")
            lines.append(f"  {content}")
            user_idx += 1
        else:
            lines.append("\n[ASSISTANT]")
            lines.append(f"  {content}")

    lines.append("")
    lines.append("-" * 80)
    lines.append(f"LABELER says: {sample['labeler_label']}")
    lines.append(f"Rationale:    {sample['labeler_rationale']}")
    lines.append("=" * 80)
    return "\n".join(lines)


def run_audit(audit_path: Path) -> None:
    samples = load_jsonl(audit_path)
    n_total = len(samples)
    n_done = sum(1 for s in samples if s.get("human_label") is not None)
    print(f"\nAudit progress: {n_done}/{n_total} done. Type 'h' anytime for help.\n")

    for i, sample in enumerate(samples):
        if sample.get("human_label") is not None:
            continue
        print(render_sample(sample))
        print(f"\n[Sample {i + 1}/{n_total}]")

        while True:
            resp = input("verdict (y [note] / n LABEL [note] / s / q / h): ").strip()
            if resp == "" or resp.lower() == "y" or resp.lower().startswith("y "):
                sample["human_label"] = sample["labeler_label"]
                if resp.lower().startswith("y "):
                    sample["human_note"] = resp[2:].strip()
                break
            if resp.lower() == "h":
                print("  y or <enter>      → agree with labeler")
                print("  y <note>          → agree, with inline note")
                print("  n none|somewhat|strongly [optional note]  → disagree")
                print("  s                 → skip (revisit later)")
                print("  q                 → quit and save progress")
                continue
            if resp.lower() == "s":
                break
            if resp.lower() == "q":
                save_jsonl(samples, audit_path, append=False)
                print(f"\nSaved progress to {audit_path}. Resume with same command.")
                return
            if resp.lower().startswith("n"):
                parts = resp.split(maxsplit=2)
                if len(parts) >= 2 and parts[1].lower() in ("none", "somewhat", "strongly"):
                    sample["human_label"] = parts[1].lower()
                    if len(parts) == 3:
                        sample["human_note"] = parts[2]
                    else:
                        note = input("note (optional, press enter to skip): ").strip()
                        sample["human_note"] = note
                    break
                print("  Format: n <none|somewhat|strongly> [optional inline note]")
                continue
            print("  Unknown command. Type 'h' for help.")

        save_jsonl(samples, audit_path, append=False)

    save_jsonl(samples, audit_path, append=False)
    print("\nAudit complete.\n")
    score(audit_path)


def score(audit_path: Path) -> None:
    samples = load_jsonl(audit_path)
    done = [s for s in samples if s.get("human_label") is not None]
    if not done:
        print("No completed audits yet. Run without --score-only to start.")
        return

    agree = sum(1 for s in done if s["human_label"] == s["labeler_label"])
    pct = agree / len(done)
    print(f"\nAudit results: {agree}/{len(done)} agreement ({pct:.0%})\n")

    disagreements = [s for s in done if s["human_label"] != s["labeler_label"]]
    if disagreements:
        print(f"{len(disagreements)} disagreements:\n")
        breakdown = Counter((s["labeler_label"], s["human_label"]) for s in disagreements)
        for (lab, hum), n in breakdown.most_common():
            print(f"  labeler={lab:>8s} → you={hum:<8s}  ({n})")
        print()
        for s in disagreements:
            print(
                f"- seed {s['seed_id'][:8]}, turn {s['turn_index']}: "
                f"labeler={s['labeler_label']}, you={s['human_label']}"
            )
            if s.get("human_note"):
                print(f"    note: {s['human_note']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit_path", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--input_path", type=Path, default=DEFAULT_LABELED)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reset", action="store_true",
                        help="Discard any existing audit and start fresh.")
    parser.add_argument("--score-only", action="store_true",
                        help="Print agreement summary, don't enter audit loop.")
    args = parser.parse_args()

    if args.score_only:
        score(args.audit_path)
        return

    if args.reset and args.audit_path.exists():
        args.audit_path.unlink()
        print(f"Reset: deleted {args.audit_path}")

    if not args.audit_path.exists():
        samples = sample_turns(args.input_path, args.n, args.seed)
        save_jsonl(samples, args.audit_path, append=False)
        print(f"Sampled {len(samples)} turns -> {args.audit_path}")

    run_audit(args.audit_path)


if __name__ == "__main__":
    main()
