"""Generate a static HTML page for browsing B4 steering outputs side-by-side.

Layout: per held-out conversation, show the prefix at top, then a row of
columns (one per α) with the steered assistant response. Each response
is annotated with:
  - Sonnet-scored directiveness / hedging / compliance (0-10)
  - Mechanical metrics (length, hedge count, directive count)
  - Keyword highlighting:
    - DEFERENCE markers (red): "I'll do whatever", "you decide", "your call",
      "tell me what to do", "I trust"
    - DIRECTIVE markers (green): "you should", "I recommend", "the answer is", "do this"
    - HEDGE markers (yellow): "might", "could", "perhaps", "consider", "depends"
    - COMPLIANCE markers (blue): "happy to help", "let me know", "whatever you want", "up to you"

Run:
    uv run python -m auth_projection.build_steering_html
"""

import argparse
import html
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from utils.file_utils import load_jsonl


HIGHLIGHT_PATTERNS = {
    "deference": [
        r"\bI(?:'ll| will) do whatever\b",
        r"\byou decide\b",
        r"\byour call\b",
        r"\btell me what (?:to do|you think)\b",
        r"\bI trust\b",
        r"\byou'?re the (?:expert|boss)\b",
        r"\bjust (?:tell|pick|say)\b",
        r"\bI'll go with (?:whatever|your)\b",
        r"\bI defer to\b",
        r"\bup to you\b",
        r"\bwhatever you (?:say|want|think|choose)\b",
    ],
    "directive": [
        r"\byou should\b",
        r"\byou ought to\b",
        r"\byou need to\b",
        r"\byou must\b",
        r"\bI recommend\b",
        r"\bI suggest\b",
        r"\bI'?d (?:recommend|suggest|say)\b",
        r"\bgo with\b",
        r"\bthe answer is\b",
        r"\bthe best (?:option|choice|approach)\b",
        r"\bdefinitely\b",
        r"\babsolutely\b",
        r"\bclearly\b",
    ],
    "hedge": [
        r"\bmight\b", r"\bcould\b", r"\bperhaps\b", r"\bmaybe\b", r"\bpossibly\b",
        r"\bconsider\b", r"\bI think\b", r"\bhave you (?:thought|considered)\b",
        r"\bit (?:depends|really depends)\b", r"\bone option\b", r"\bone approach\b",
        r"\bworth considering\b", r"\bsomething to (?:think about|consider)\b",
    ],
}


def highlight_text(text: str) -> str:
    """Replace pattern matches with colored span tags. Operate on escaped text
    so injection isn't an issue, then mark matches by re-finding them."""
    escaped = html.escape(text)
    # Replace patterns in order: longer/specific first to avoid double-coloring
    for klass, patterns in HIGHLIGHT_PATTERNS.items():
        for pat in patterns:
            escaped = re.sub(
                pat,
                lambda m: f'<span class="hl-{klass}">{m.group(0)}</span>',
                escaped,
                flags=re.IGNORECASE,
            )
    # Convert paragraph breaks
    escaped = escaped.replace("\n\n", "</p><p>").replace("\n", "<br/>")
    return f"<p>{escaped}</p>"


def count_markers(text: str) -> Dict[str, int]:
    counts = {}
    for klass, patterns in HIGHLIGHT_PATTERNS.items():
        n = sum(len(re.findall(p, text, re.IGNORECASE)) for p in patterns)
        counts[klass] = n
    return counts


def render_prefix_html(conv: Dict, last_user_idx: int) -> str:
    """Render conversation prefix through last_user_idx as HTML."""
    parts = []
    user_count = 0
    for t in conv["conversation"]:
        role = t["role"]
        content_html = html.escape(t["content"]).replace("\n", "<br/>")
        parts.append(f'<div class="msg msg-{role}"><div class="msg-role">{role.upper()}</div><div class="msg-content">{content_html}</div></div>')
        if role == "user":
            if user_count == last_user_idx:
                break
            user_count += 1
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steering_path", type=Path,
                        default=Path("auth_projection/data/v3b_steering_real_conversations.json"))
    parser.add_argument("--scored_path", type=Path,
                        default=Path("auth_projection/data/v3b_steering_responses_scored.jsonl"))
    parser.add_argument("--convs_path", type=Path,
                        default=Path("auth_projection/data/v1_labeled.jsonl"))
    parser.add_argument("--output_path", type=Path,
                        default=Path("auth_projection/data/v3b_steering_viewer.html"))
    args = parser.parse_args()

    samples = json.load(open(args.steering_path))
    scored = {}
    if args.scored_path.exists():
        for r in load_jsonl(args.scored_path):
            scored[(r["seed_id"], r["alpha"])] = r
    convs = load_jsonl(args.convs_path)
    convs_by_seed = {c["seed_id"]: c for c in convs}

    # Group steering samples by seed_id, keep alpha order
    by_seed = defaultdict(list)
    for s in samples:
        by_seed[s["seed_id"]].append(s)
    for seed in by_seed:
        by_seed[seed].sort(key=lambda x: x["alpha"])

    # Get sorted list of all alphas (assumed same set per conv)
    all_alphas = sorted({s["alpha"] for s in samples})

    css = """
    <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 1800px; margin: 1em auto; padding: 0 1em; color: #1a1a1a; }
    h1 { margin-top: 0.5em; }
    .summary-box { background: #f5f5f5; padding: 1em 1.5em; border-radius: 6px; margin: 1em 0; }
    .conv-block { margin: 2em 0; padding-bottom: 1em; border-bottom: 2px solid #ddd; }
    .conv-header { background: #fff8e7; padding: 0.6em 1em; border-radius: 4px; margin-bottom: 0.6em; font-size: 0.9em; }
    .conv-header b { color: #555; }
    .prefix-block { background: #fafafa; padding: 0.5em 1em; border-radius: 4px; margin-bottom: 1em; max-height: 400px; overflow-y: auto; font-size: 0.9em; }
    .msg { margin: 0.5em 0; }
    .msg-role { font-size: 0.7em; font-weight: 700; color: #888; letter-spacing: 0.04em; }
    .msg-content { padding: 0.4em 0.6em; border-radius: 4px; margin-top: 0.2em; }
    .msg-user .msg-content { background: #e7f0ff; }
    .msg-assistant .msg-content { background: #f0f0f0; }
    .alpha-row { display: grid; grid-template-columns: repeat(var(--n-alphas), 1fr); gap: 0.6em; }
    .alpha-cell { background: white; border: 1px solid #ddd; border-radius: 4px; padding: 0.6em 0.8em; font-size: 0.85em; }
    .alpha-cell.alpha-zero { background: #fffacd; border-color: #c0a040; }
    .alpha-cell.alpha-pos { background: #ffe7e7; border-color: #d09090; }
    .alpha-cell.alpha-neg { background: #e7f0e7; border-color: #80c080; }
    .alpha-label { font-weight: 700; font-size: 0.85em; }
    .alpha-meta { font-size: 0.7em; color: #777; margin-top: 0.3em; }
    .scores { display: flex; gap: 0.4em; font-size: 0.7em; flex-wrap: wrap; margin: 0.4em 0; }
    .score { background: #eee; padding: 0.1em 0.4em; border-radius: 3px; }
    .completion p { margin: 0.4em 0; }
    .summary-line { font-style: italic; color: #555; font-size: 0.8em; margin-top: 0.5em; }
    .hl-deference { background: #ffcccc; padding: 1px 3px; border-radius: 3px; }
    .hl-directive { background: #ccffcc; padding: 1px 3px; border-radius: 3px; }
    .hl-hedge { background: #ffffcc; padding: 1px 3px; border-radius: 3px; }
    .hl-compliance { background: #cce5ff; padding: 1px 3px; border-radius: 3px; }
    .legend { display: flex; gap: 1em; flex-wrap: wrap; margin: 0.5em 0; font-size: 0.85em; }
    .legend-item { padding: 2px 6px; border-radius: 3px; }
    table { border-collapse: collapse; margin: 1em 0; }
    th, td { padding: 0.3em 0.6em; border: 1px solid #ccc; text-align: right; font-size: 0.85em; }
    th { background: #eee; }
    </style>
    """

    # Aggregate stats by alpha for the summary table
    agg = defaultdict(lambda: {"n": 0, "directiveness": [], "hedging": [], "compliance": [],
                                "len_chars": [], "deference_n": [], "directive_n": [], "hedge_n": []})
    for s in samples:
        a = s["alpha"]
        agg[a]["n"] += 1
        agg[a]["len_chars"].append(len(s["completion"]))
        m = count_markers(s["completion"])
        agg[a]["deference_n"].append(m["deference"])
        agg[a]["directive_n"].append(m["directive"])
        agg[a]["hedge_n"].append(m["hedge"])
        sc = scored.get((s["seed_id"], a))
        if sc:
            agg[a]["directiveness"].append(sc["directiveness"])
            agg[a]["hedging"].append(sc["hedging"])
            agg[a]["compliance"].append(sc["compliance"])

    def fmt_mean(xs):
        if not xs:
            return "—"
        return f"{sum(xs)/len(xs):.2f}"

    summary_rows = []
    for a in all_alphas:
        data = agg[a]
        summary_rows.append(
            f"<tr><td><b>α={a:+.1f}</b></td>"
            f"<td>{data['n']}</td>"
            f"<td>{fmt_mean(data['len_chars'])}</td>"
            f"<td>{fmt_mean(data['directiveness'])}</td>"
            f"<td>{fmt_mean(data['hedging'])}</td>"
            f"<td>{fmt_mean(data['compliance'])}</td>"
            f"<td>{fmt_mean(data['deference_n'])}</td>"
            f"<td>{fmt_mean(data['directive_n'])}</td>"
            f"<td>{fmt_mean(data['hedge_n'])}</td>"
            f"</tr>"
        )

    summary_table = f"""
    <table>
      <tr>
        <th>α</th><th>n</th><th>mean len (chars)</th>
        <th>Sonnet directiveness</th><th>Sonnet hedging</th><th>Sonnet compliance</th>
        <th>defer markers</th><th>directive markers</th><th>hedge markers</th>
      </tr>
      {"".join(summary_rows)}
    </table>
    """

    # Per-conversation rendering
    conv_blocks = []
    for seed_id, items in by_seed.items():
        conv_full = convs_by_seed.get(seed_id, {})
        last_user_idx = items[0].get("last_user_index", 0)
        target_tier = items[0].get("target_tier", "?")
        twin_kind = items[0].get("lexical_twin_kind", "match") or "match"
        topic = items[0].get("topic", "?")
        last_user_label = items[0].get("last_user_label", "?")

        prefix_html = render_prefix_html(conv_full, last_user_idx)

        cells = []
        for item in items:
            a = item["alpha"]
            cls = "alpha-zero" if a == 0 else ("alpha-pos" if a > 0 else "alpha-neg")
            sc = scored.get((seed_id, a))
            mech = count_markers(item["completion"])
            scores_html = ""
            if sc:
                scores_html = (
                    f'<div class="scores">'
                    f'<span class="score">📈 directive {sc["directiveness"]}/10</span>'
                    f'<span class="score">🤔 hedging {sc["hedging"]}/10</span>'
                    f'<span class="score">🙇 compliance {sc["compliance"]}/10</span>'
                    f'</div>'
                )
            mech_html = (
                f'<div class="scores">'
                f'<span class="score">len {len(item["completion"])}</span>'
                f'<span class="score">defer-words {mech["deference"]}</span>'
                f'<span class="score">direct-words {mech["directive"]}</span>'
                f'<span class="score">hedge-words {mech["hedge"]}</span>'
                f'</div>'
            )
            summary_line = f'<div class="summary-line">{html.escape(sc["summary"])}</div>' if sc else ""
            cells.append(
                f'<div class="alpha-cell {cls}">'
                f'<div class="alpha-label">α = {a:+.1f}</div>'
                f'{mech_html}{scores_html}'
                f'<div class="completion">{highlight_text(item["completion"])}</div>'
                f'{summary_line}'
                f'</div>'
            )

        conv_blocks.append(f"""
        <div class="conv-block">
          <div class="conv-header">
            <b>seed:</b> {seed_id[:12]}…
            &nbsp;|&nbsp; <b>topic:</b> {html.escape(topic)}
            &nbsp;|&nbsp; <b>target tier:</b> {target_tier}
            &nbsp;|&nbsp; <b>twin kind:</b> {twin_kind}
            &nbsp;|&nbsp; <b>label of last user turn:</b> {last_user_label or "?"}
          </div>
          <details>
            <summary><b>Conversation prefix</b> (click to expand)</summary>
            <div class="prefix-block">{prefix_html}</div>
          </details>
          <div class="alpha-row" style="--n-alphas: {len(items)};">
            {"".join(cells)}
          </div>
        </div>
        """)

    # Final HTML
    n_alphas = len(all_alphas)
    html_body = f"""
    <!DOCTYPE html>
    <html><head><title>B4 Steering Viewer</title>{css}</head><body>
    <h1>Real-conversation full-position steering — Llama 3.1 8B at L14</h1>

    <div class="summary-box">
      <p><b>What you're looking at.</b> Each block below is one held-out test conversation. The prefix (collapsed by default) shows the full conversation through the last user turn. Below that is a row of columns, one per steering strength α. Each column shows the assistant response that was generated when α·v was injected at every token during generation, with all sampling parameters (seed, temperature) locked across alphas. The only thing that changes is α.</p>
      <p><b>Reading guide:</b> α=0 (yellow) is baseline — no steering. Negative α (green) pushes the model toward "user is independent" representation. Positive α (red) pushes toward "user is deferring." If steering works, you should see a monotone shift across columns.</p>
      <div class="legend">
        <span class="legend-item hl-deference">deference markers</span>
        <span class="legend-item hl-directive">directive markers</span>
        <span class="legend-item hl-hedge">hedge markers</span>
      </div>
    </div>

    <h2>Aggregate statistics by α</h2>
    {summary_table}
    <p style="font-size: 0.85em; color: #666;">If steering works, the directiveness column should monotonically increase from negative to positive α and hedging should decrease (or compliance — assistant ceding back to user — should increase).</p>

    <h2>Per-conversation comparisons</h2>
    {"".join(conv_blocks)}

    </body></html>
    """

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        f.write(html_body)
    print(f"Saved viewer -> {args.output_path}")
    print(f"Open in browser: file://{args.output_path.absolute()}")


if __name__ == "__main__":
    main()
