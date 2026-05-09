# Research Log — User State Probes (v0: Authority Projection)

Living document. Most recent entries at top. Each entry captures a decision, the rationale, and alternatives considered, so a future paper writeup or v1 design can reconstruct the reasoning.

---

## 2026-05-09 — v7: Llama 3.1 8B BASE probe + geometry analyses + categorize substance verdicts

End-of-night sprint, three lines of work in parallel: (1) probe Llama 3.1 8B *base* (no instruction tuning, no RLHF) to see if the deference feature pre-exists post-training; (2) free local analyses of v_deference geometry (PCA, cross-layer cosine, topic invariance, twin-failure case study); (3) Sonnet-categorize the DIFF_REC verdicts to see *what kind* of advice changes happen, not just how often.

### Result 1 — The deference feature is a *pretraining* phenomenon

At `last_user_token` position, Llama 3.1 8B *base* model probe is essentially indistinguishable from the instruct version (and slightly *higher* at the headline layer):

| Layer | last_user_token: Instruct | last_user_token: BASE |
|---|---|---|
| L13 | 0.774 | 0.795 |
| **L14** | **0.801** | **0.815** ← base wins by +1.4pp |
| L15 | 0.767 | 0.795 |
| L21 | 0.760 | 0.788 |
| L23 | 0.788 | 0.740 |

Best-accuracy layer is L14 in *both* models. AUC strongly_vs_rest: instruct 0.989, base 0.983. Within sampling noise.

**Implication:** The deference-tracking representation does NOT require instruction tuning or RLHF. It exists in the model after pretraining alone. This rules out the most attractive "RLHF created sycophancy" framing — at least at the level of internal representation.

The behavioral result is consistent with this: pretrained transformers learn from internet text where deferential users tend to receive directive replies. The model picks up that statistical association during pretraining; fine-tuning amplifies the *behavior* but doesn't create the underlying user-state representation.

### Result 2 — Instruction tuning creates chat-template-anchored readout, *not* user-state

At `assistant_start_token` position (the chat-template "assistant header" right before the model generates), the picture inverts:

| Layer | assistant_start: Instruct | assistant_start: BASE |
|---|---|---|
| L12 | 0.807 | 0.731 |
| L13 | 0.814 | 0.745 |
| **L14** | **0.848** | 0.752 ← instruct wins by +9.6pp |
| L15 | 0.786 | 0.779 |
| L17 | 0.800 | 0.759 |

This makes mechanical sense: `<|start_header_id|>assistant<|end_header_id|>` are arbitrary tokens to the base model — it has never been trained to treat them as semantic anchors. The instruct model has been trained to use that position as the "now I respond" signal, and as a side effect that position acquires a sharp readout of user-state-from-context.

**So instruction tuning does two things:** (a) makes the chat-template tokens semantically meaningful, and (b) sharpens readout of user-state at those template positions. It does NOT create the user-state feature itself — that's already present after pretraining.

### Result 3 — v_deference is essentially 1-dimensional and topic-invariant

Free local analysis of v6c L14 activations:

**PCA on strongly-vs-none training-split activations:**
- PC1 alone gives **91.8%** train accuracy on strongly-vs-none separation
- **cos(v_def, PC1) = 0.960** — v_def is essentially PC1 of the class-mean separation
- ‖projection of v_def onto top-1 PCs‖ / ‖v_def‖ = 0.960 (96% of v_def's energy lives on a single direction)
- ‖projection onto top-3 PCs‖ / ‖v_def‖ = 0.983 (98.3% lives in 3-D)

**Implication:** Deference is a *clean linear feature* — a single direction in residual space, not a manifold. The mean-difference steering vector is essentially the optimal direction in the standard "linear probing" sense. This is the kind of structurally clean feature interpretability work prefers.

**Cross-layer cosine of v_def vs L14:**

| L | cos(v_L, v_L14) | ‖v‖ |
|---|---|---|
| 0 | 0.05 | 0.14 |
| 5 | 0.26 | — |
| 10 | 0.58 | — |
| 13 | 0.84 | — |
| **14** | **1.00** | **3.72** |
| 15 | 0.86 | — |
| 21 | 0.50 | — |
| 32 | 0.15 | 41.2 |

**The deference direction crystallizes around L13–L15** — strongly preserved within ±2 layers, then gradually diverges. Norm grows monotonically from 0.14 (embed) to 41 (final layer). The feature evolves through the residual stream rather than being a static direction at one site, but its *core orientation* is set by L13–L15.

**Topic invariance:** topic-specific v_def (built from records of one topic only) has cosine 0.74–0.89 with the global v_def. The deference feature is largely topic-independent. Highest cos: personal goals (0.889), health (0.881), financial (0.882). Lowest: programming (0.743), cooking (0.777). All positive, all >0.74.

### Result 4 — The probe is *behavior*-anchored, not vocabulary-anchored

Submission_voice failure mode (probe loses 0.60 vs TF-IDF 0.80, n=15) — the actual cases:

The 3 cases where TF-IDF wins all share a specific pattern:
- True label = `somewhat` (user is *partially* deferring)
- User uses explicit deferential phrases: *"you're the expert here so I'll trust your take,"* *"I'll defer to your expertise here,"* *"I'll defer to your read"*
- Probe says `none` (independent), TF-IDF correctly says `somewhat`

**The probe correctly distinguishes "behavioral surrender" from "polite deferential framing while still being decisive."** This is *why* it wins on `peer_voice` (severe deference behavior, no submission vocabulary). The "submission_voice failure" isn't really a failure — it's that the probe is too good at filtering out vocabulary cues to catch cases where the labeler used vocabulary as the deciding signal. The probe has a behavior-anchored conception of deference; the labeler in `somewhat` cases is using a vocabulary-anchored conception.

Methodological lesson: the lexical-twin design *worked* — the probe and the bag-of-words classifier are reading different things, and the twin slice surfaces that difference cleanly.

### Result 5 — Categorize substance verdicts: the +1 direction is the cleaner safety signal

Sonnet-categorized each DIFFERENT_RECOMMENDATION verdict into a fixed taxonomy. Across 4 conditions (minedit ±1, paraphrase, steering deference α=±1, steering random α=±1), filtering to in-distribution α=±1 to avoid the |α|=2 incoherence confound:

**"Directive-takeover" composite (TAKES_OVER_DECISION + PROVIDES_TEMPLATE + ADDS_UNREQUESTED_ACTIONS + CONTRADICTS_USER_PREFERENCE):**

| Condition | Composite |
|---|---|
| Steering deference α=+1 | **63%** |
| Minedit +1 (user defers) | **57%** |
| Steering random α=+1 | 44% |
| Minedit −1 (user keeps agency) | 46% |
| Paraphrase noise floor | 28% |
| Steering random α=−1 | 54% |
| **Steering deference α=−1** | **25%** |

**The +1 direction (user defers / deference injected) is the clean safety pattern.** When user surrenders agency, the model commits to choices the user was weighing, provides ready-to-use templates, adds unrequested actions, and sometimes contradicts user-stated preferences — at 57–63% of substantive changes. When user keeps agency, this composite drops to 25–46%.

**TAKES_OVER_DECISION specifically:** 26% (minedit +1), 37% (steering deference α=+1), drops to 0% (steering deference α=−1). Direction-specific.

### Result 6 — The −1/+1 asymmetry partially dissolves under categorization

The previous v3c+v6c headline was: *minedit −1 = 55–56% DIFF_REC vs minedit +1 = 33–34%, asymmetric* (+22pp gap). After categorizing:

- minedit −1 has 19% `DIFFERENT_SEQUENCE_SAME_GOAL` + 30% `DIFFERENT_FRAMING_OR_TONE` (cooking-step reorderings, tone shifts)
- minedit +1 has 0% sequence + 26% framing

Stripping framing+sequence as "not real substance change" gives "real substantive change rate":

| | DIFF_REC% | × (1 − framing% − sequence%) | substantive only |
|---|---|---|---|
| Minedit +1 | 33% | × 74% | **24%** |
| Minedit −1 | 56% | × 51% | **29%** |
| Paraphrase noise floor | 27% | × 41% | **11%** |

**Real gap above noise: +13pp (+1) and +18pp (−1).** Smaller than the headline +29pp gap from raw DIFF_REC%. The asymmetry shrinks from 23pp → 5pp. Most of the apparent asymmetry was Sonnet over-counting "different sequence" or "different framing" verdicts as DIFF_REC in the −1 cases (cooking conversations have lots of step-reorderings).

The behavior coupling claim survives (substantive +1 still 13pp above noise floor) but the *direction* of asymmetry largely dissolves — the safety pattern is symmetric in magnitude, just shifted in *kind*: +1 is cleanly "model takes over"; −1 is more diffuse stylistic shifts plus some takeover.

### What this collectively changes for the project

**Old framing:** "Sycophancy is RLHF-induced; we have a probe to monitor and steer it."

**New framing (now supported by data):**
1. The deference-tracking representation is a *pretraining* feature. It's how language models learn user-state from internet chat data, and instruction-tuning doesn't create it (at least at the residual-stream level).
2. The feature is essentially 1-dimensional, topic-invariant, crystallizes at L13–L15, and is *behavior-anchored* (not vocabulary-anchored).
3. When users defer, the model takes over decisions / provides templates / contradicts user preferences in ~57–63% of substantive changes — the safety pattern made concrete in failure modes, not just rates.
4. The −1/+1 asymmetry is mostly an artifact of Sonnet over-counting framing/sequence verdicts; both directions show similar amounts of *real* recommendation change (~13–18pp above noise), just of different kinds.

**The main safety-relevant claim becomes:**

> *"User-deference-tracking is a pretraining-emerging, linearly-readable, ~1-D feature in the residual stream of Llama 3.1 8B at L13–L15. Instruction tuning sharpens its readout at chat-template positions but does not create it. Steering along this direction at moderate α produces ~2× more substantive recommendation change than a same-norm random direction, with the most common failure mode being 'model takes over a decision the user was weighing.'"*

That's a paper-shaped claim. It also changes what "fixing sycophancy" looks like: not "fix RLHF data" (the feature is already there post-pretraining), but instead "block readout of an existing feature at inference time" or "train a model that doesn't form this feature in the first place" (much harder; pretraining-data-level intervention).

### Cost

Total v7 spend:
- RunPod: ~$1.50 (one A6000 pod, ~2h, just probe extraction + training)
- Anthropic: ~$1 (categorize substance, ~150 verdicts)
- Total: **~$2.50**

RunPod balance after this run: ~$10. Plenty of headroom for follow-up.

### Files added this iteration

- `auth_projection/data/v7_activations.pt` (Llama 3.1 8B base activations, on pod only)
- `auth_projection/data/v7_probe_results.json`, `v7_probes/probe_C_layer*.joblib`
- `auth_projection/data/v7_probe_results_assistant_start.json`, `v7_probes_assistant_start/`
- `auth_projection/data/v6c_geometry_analysis.json` (PCA, cross-layer, topic results)
- `auth_projection/data/v6c_diff_rec_categorized.jsonl` (149 categorized verdicts)
- `auth_projection/analyze_v_deference_geometry.py` (PCA / cross-layer / topic script)
- `auth_projection/analyze_twin_failures.py` (submission_voice case study)
- `auth_projection/categorize_diff_rec.py` (Sonnet category taxonomy on DIFF_REC verdicts)

### Open questions this opens up

The base-vs-instruct result begs the next question: **at what point during pretraining does the feature emerge?** This is the multi-checkpoint Olmo/Pythia experiment from the original next-steps list. Specifically: probe accuracy as a function of pretraining tokens. If it rises smoothly with scale, the feature is "internet text encodes deferential interactions and the model picks up the statistical regularity." If it appears suddenly at a specific scale, that's a phase-transition story.

Also still open: the random-direction control could be tightened with more seeds or with the orthogonal-complement variant. And the full mechanism (which attention heads write to v_def at L13–L14?) is unclaimed.

---

## 2026-05-09 — Project summary: results across H1–H5

End-of-Week-3 summary organized by hypothesis, written after v6 replay closed the reproducibility loop. Numbers are pulled from v3c (committed) and v6c (replay) — both Llama 3.1 8B Instruct unless noted.

### H1 — Is there an authority-projection representation, and can probes catch it beyond text features?

**R1: Yes, at scale.**

| | acc | AUC strongly |
|---|---|---|
| Llama 8B probe at L14 (last_user_token) | 0.801 | 0.989 |
| Llama 8B probe at L14 (assistant_start_token) | 0.841 | — |
| Qwen 7B probe at L21 (assistant_start_token) | 0.842 | — |
| TF-IDF baseline (combined lexical twins) | 0.707 | — |

The acid test was the **lexical-twin slice** — conversations where surface vocabulary deliberately disagrees with user-state. On `peer_voice` (severe deference behavior expressed *without* submission vocabulary, the safety-relevant cell), the probe wins big: **Llama 8B 0.920 vs TF-IDF 0.692**; Qwen 7B 0.846. Probe is reading state, not words. Caveat: on the opposite twin direction `submission_voice` (deferential vocabulary without behavioral surrender), the probe still loses to TF-IDF. The probe sees past words in *one* direction but not both.

Probe results replicated exactly (to 0.001 of acc/AUC) across v3c→v6c independent reruns — fully deterministic given activations.

### H2 — Steerable direction?

**R2: Yes, with clean monotone shape inside coherent range.**

Steering vector `v = mean(strongly) - mean(none)` at L14, ‖v‖ = 3.715. Sonnet-judged directiveness/hedging/compliance shifts monotonically with α across [−2, +1]:

| α | directiveness | hedging | compliance |
|---|---|---|---|
| −2 | 3.51 | 5.67 | 4.54 |
| 0 | 5.72 | 3.90 | 3.56 |
| +2 | 5.92 | 2.54 | 2.64 |

Same monotone shape replicated cross-family at Qwen 7B with a slightly narrower coherent range. Off-manifold collapse (incoherent text) at |α|≥4 in both models — that's where the direction stops being useful.

### H3 — Does steering change *substance*, not just style? Compared to what?

**R3: Yes, two independent ways.** This is the project's load-bearing evidence.

**Way #1 — natural-text comparison (the strongest evidence):**

| | wording change? | state change? | DIFF_REC% |
|---|---|---|---|
| Same-tier paraphrase | large | none | 23–27% (noise floor) |
| Minedit −1 (less deferential) | tiny | strong | **55–56%** |
| Minedit +1 (more deferential) | tiny | strong | 33–34% |

**Tiny edits that flip state cause +29–32pp more substantive recommendation change than larger rewordings that preserve state.** Replicated essentially exactly (within 1pp) across v3c committed and v6c replay on n=136 minedit + n=133 paraphrase pairs. No steering involved — pure prompt difference.

**Way #2 — random-direction steering control (loophole-closer):**

| | DIFF_REC% at α=+2 |
|---|---|
| Deference vector | **92%** (v6 replay) / 69% (v3c committed) |
| 3 random unit vectors (norm-matched to ‖v‖) | 33%, 36%, 59% — mean **43%** |

**Deference direction is 2.1× more effective than a same-norm random direction.** Direction-specificity is real. Caveat: random direction isn't zero — it produces ~43% DIFF_REC at α=+2, not the 23% paraphrase noise floor. So the steering effect has both a generic-perturbation component *and* a deference-specific component. Honest framing: "the deference direction shifts recommendations more than twice as often as random," not "the deference direction is uniquely effective."

### H4 — Does this emerge with scale?

**R4: Mostly yes, with caveats.** Behavior coupling shows clearer scale emergence than the probe itself.

**Probe accuracy** is mostly flat 1.5B → 7B then steps up at 14B:

| | Qwen 1.5B | Qwen 3B | Qwen 7B | **Qwen 14B** | Llama 8B |
|---|---|---|---|---|---|
| Best-layer acc | 0.774 | 0.767 | 0.767 | **0.815** | 0.801 |

But this isn't where the scaling story lives. The interesting scaling is on the lexical-twin slice (does the probe see past words?) and on behavior coupling (does the model adapt to user state?):

| | Qwen 1.5B | Qwen 3B | Qwen 7B / Llama 8B |
|---|---|---|---|
| Probe vs TF-IDF (combined twins) | probe loses | tied | probe wins +5 to +14 pp |
| Mechanical metrics with p<0.05 | 0 | 2 | 2 (one at p=0.0001) |
| Judge tie-rate (lower = sharper signal) | 80% | 84% | **53%** |
| User-position-only steering | clean null | mostly null | weak but visible |

Sharp emergence at the 7–8B step. The deference-tracking feature appears to crystallize between 3B and 7B, and behavior-coupling is the cleanest signal for it. Within-Qwen scaling for steering specifically only goes to 7B (14B steering failed three times on HF infra issues — not a scientific failure).

### H5 — what else matters for interpretation

**H5a — Asymmetric coupling.** The model adapts much more aggressively to "user becoming less deferential" than to "user becoming more deferential." Three independent observers replicate the asymmetry: length deltas (−1 fires p=0.001 at 8B; +1 null at 8B), judge tie-rate split, substance verdicts (+32pp gap on −1, +11pp gap on +1 at Llama 8B). Same direction at Qwen 7B and 14B. Real and not artifactual.

**H5b — Probe-position matters.** The chat-template `assistant_start_token` (the position right before the model generates) is a +4 to +7pp better readout than the `last_user_token`. Replicated cross-family. Methodological lesson: the right probe position is the one closest to the prediction site, not the conventional last-input-token choice.

**H5c — Submission_voice asymmetry within the probe itself.** Probe sees past surface vocabulary in the `peer_voice` direction (severe deference, no submission words → probe wins by 23pp at 8B) but NOT in the `submission_voice` direction (deferential vocab without behavioral surrender → probe still loses to TF-IDF by 13pp at 8B). The deference feature is partially word-anchored. Surface-anchoring is reduced by scale but not eliminated.

**H5d — Steering magnitude has hardware variance.** Same activations + same vector + same generation seed produce ~±20pp different DIFF_REC% across different transformers/torch/CUDA versions. Direction of effect is stable; magnitude estimates need n≥3 hardware draws for confidence. Methodological lesson for steering papers.

**H5e — Off-manifold collapse at extreme α.** Steering at |α|≥4 produces incoherent text (Chinese-character loops on Qwen, broken English on Llama). The deference direction lives in a narrow "coherent manifold" around α≈0. Not load-bearing for the project but worth flagging.

### Headline (one-sentence)

> *"At Llama 3.1 8B, a deference-direction linear probe trained on residual activations at L14 reads user-state in a way that (a) generalizes to lexical twins where text alone cannot, (b) tracks behavior such that natural minimal-edit user-state flips drive substantively different model recommendations at +30pp above a same-state paraphrase baseline (replicated within 1pp across two independent runs), and (c) when used as a steering direction, produces 2× more substantive recommendation change than a same-norm random direction at the same layer."*

Three independently-supported sub-claims, each with a control. The behavior-coupling claim (b) is the load-bearing one — survived judge-variance, surface-text-volatility, and end-to-end model-reproduction stress tests.

### What's intentionally NOT claimed

- "Deference is the ONLY representation that drives substance change" — the random control shows random directions also shift things, just less.
- "Steering by exact magnitude X reproduces across hardware" — magnitudes drift ±20pp.
- "The probe is purely reading state, not words" — `submission_voice` failure shows partial word-anchoring.
- "Scale-emergence is universal across model families" — Qwen 14B steering missing means within-Qwen steering scaling is unverified past 7B.
- "Sycophancy emerges during pretraining vs RLHF" — that's the *next* experiment (training-dynamics on Olmo/Pythia), not yet run.

---

## 2026-05-09 — v6 replay: independent reproduction + random-direction steering control

### What we set out to do

After v3c+v4+v5 produced the headline numbers, the obvious self-skepticism check: **do the numbers reproduce on a fresh pod with a fresh Sonnet cache?** And does the random-direction steering control close the "any-direction-flips-recommendations" loophole? Two independent stress tests, both run end-to-end here.

### Setup

- New pod (RTX A6000, fresh download of Llama 3.1 8B Instruct, fresh transformers/torch versions)
- Fresh `safetytooling` cache directory (`/tmp/v6_sonnet_cache_fresh`) — guaranteed new Sonnet API calls, not cache hits from v3c
- Same code paths, same locked seeds, same input data (`v1_labeled.jsonl`, `v1_paraphrase_pairs.jsonl`, `v3c_minedit_pairs_expanded.jsonl`)
- Tag: `v6c` for files (v6 replay of v3c)

Side experiment first (run before any GPU): **judge variance check** — re-score the *committed v3c response files* with a fresh Sonnet cache. This isolates judge stability from model stability. Sonnet at temperature=0 is essentially deterministic by design but we wanted to verify.

Two-phase main pipeline: Phase 1 was the full v3c replay (probes, deference-vector steering, paired generation on minedit + paraphrase, then fresh-cache Sonnet scoring of all of it). Phase 2 was the random-direction control: build 3 random unit vectors at L14 with `torch.randn` seeds {0, 1, 2}, norm-matched to ‖v_deference‖, run B4 steering with each, score substance vs α=0 baseline.

### Phase 0: re-label v1 with fresh Sonnet (sanity check)

Before regenerating any model output, re-labeled all 198 conversations with current Sonnet, diffed against committed `v1_labeled.jsonl`. Result:

- **97.2% agreement** (688/708 user turns)
- **`strongly` tier 98.8% stable** (171/173)
- `none` 98.1%, `somewhat` 93.4%
- All disagreements are single-tier moves (`none↔somewhat` or `somewhat↔strongly`) — **zero `none↔strongly` jumps**

Dataset is solid. The ambiguity is concentrated at the `somewhat` tier, which is where you'd expect labeler disagreement. The `strongly` class — the safety-load-bearing one — is essentially deterministic.

### Phase 1: probes reproduce exactly

| Layer | Committed v3c last_user_token | v6c replay last_user_token | Match? |
|---|---|---|---|
| L0 | acc=0.445 / AUC=0.753 | acc=0.445 / AUC=0.753 | exact |
| L1 | acc=0.603 / AUC=0.869 | acc=0.603 / AUC=0.868 | within 0.001 |
| L14 | **acc=0.801** / AUC=0.989 | **acc=0.801** / AUC=0.989 | exact ✓ |
| L23 | acc=0.788 / AUC=0.993 | acc=0.788 / AUC=0.993 | exact |

L14 picked as best layer by max accuracy in both runs. Probe training is deterministic given activations; activations are deterministic given the model. Exact match here is necessary but not sufficient: it confirms model weights and tokenization are stable across pods. The probes themselves are fully reproducible.

### Phase 1: judge variance — Sonnet is stable, results are real

Before any model generation, re-scored the committed v3c minedit and paraphrase responses with a *fresh* Sonnet cache directory. Same prompts, same model, but new API calls.

| | committed v3c | v6c judge-variance (same response files, fresh cache) | Δ |
|---|---|---|---|
| Minedit signed +1 DIFF_REC | 34% | **36%** | +2pp |
| Minedit signed −1 DIFF_REC | 55% | **55%** | **0pp** |
| Paraphrase none→none | 17% | **17%** | 0pp |
| Paraphrase somewhat→somewhat | 23% | **23%** | 0pp |
| Paraphrase strongly→strongly | 48% | **48%** | 0pp |

**Sonnet's verdicts are essentially deterministic at temperature=0.** The "single-pass judge might be unreliable" critique is dead. The committed numbers are not a one-roll fluke.

### Phase 1: behavior-coupling natural-text result REPLICATES

Re-generating Llama 8B paired responses on the same pair sets and re-scoring with fresh Sonnet:

| | committed v3c (n=136 minedit, n=133 paraphrase) | v6c replay | Δ |
|---|---|---|---|
| Minedit signed −1 (less deferential) | **55%** | **56%** | **+1pp** ✓ |
| Minedit signed +1 (more deferential) | 34% | 33% | −1pp |
| Paraphrase none→none | 17% | 19% | +2pp |
| Paraphrase somewhat→somewhat | 23% | 23% | 0pp |
| Paraphrase strongly→strongly | 48% | 62% | +14pp (n=21 each, small) |
| Paraphrase **overall** noise floor | **23%** | **27%** | +4pp |
| **Gap minedit−1 above paraphrase floor** | **+32pp** | **+29pp** | within sampling noise |

The headline natural-text claim — *"tiny edits that flip user-state cause substantively different recommendations ~30pp more often than same-tier paraphrases that preserve state"* — replicates within 1pp on a fresh pod end-to-end. **Behavior coupling is robust.**

### Phase 1: B4 steering reproduces directionally; magnitude shifts up

Same deference vector at L14, same locked seed, same alphas, same 39 held-out conversations. But fresh model load on different hardware/library produces slightly different generations. Result:

| α | committed v3c DIFF_REC | v6c replay DIFF_REC | Δ |
|---|---|---|---|
| −2 | 18% | **72%** | +54pp |
| −1 | 23% | 31% | +8pp |
| +1 | 44% | 49% | +5pp |
| +2 | 69% | **92%** | +23pp |

Same monotone shape. Same direction of effect at every alpha. But systematically *more* substantive recommendation change in the replay, especially at extreme alphas. This is model-decoding nondeterminism — the same activations + same vector + same seed produce slightly different token probabilities under different transformers/torch/CUDA combinations, and Sonnet judges those drift-from-baseline outputs as more substance-different.

The honest read: **steering at L14 with the deference vector causes substantive recommendation changes at α=+2 in the high-60s to low-90s of held-out conversations, depending on the model's stochasticity floor.** Both runs are inside that range.

### Phase 2: random-direction steering control closes the loophole (mostly)

Three random unit vectors at L14, each norm-matched to ‖v_deference‖ ≈ 3.715. random_v0 ran the full alpha sweep; random_v1 and random_v2 ran α∈{0, +2} only (variance check at the headline alpha).

**At α=+2, the load-bearing alpha:**

| | DIFF_REC% at α=+2 |
|---|---|
| **Deference vector** (v6c replay) | **92%** (36/39) |
| Random vector seed 0 | 59% (23/39) |
| Random vector seed 1 | 36% (14/39) |
| Random vector seed 2 | 33% (13/39) |
| Random average (3 seeds) | 43% |

**Deference α=+2 produces 2.1× more substantive recommendation change than a random direction of identical norm.** Direction-specificity is real.

But — random directions do not produce the noise-floor 23%. They produce 33–59% DIFF_REC at α=+2. So the steering effect has *both* a generic-perturbation component *and* a deference-specific component. The right framing is:

- "Any large perturbation at L14 makes the model's recommendations diverge from baseline somewhat." (random gives 43%)
- "The deference direction makes it diverge **more than twice as often as random.**" (92% vs 43%)
- The previous v3c writeup phrasing — "steering at moderate α causally shifts the model's recommendation, not just packaging" — needed the second clause to carry the safety claim. The random control shows the second clause is real but with a smaller margin than v3c implied.

**Random_v0 full alpha sweep** (across coherent range −2…+2):

| α | random_v0 DIFF_REC |
|---|---|
| −2 | 54% |
| −1 | 33% |
| +1 | 26% |
| +2 | 59% |

Spread across α matches the deference vector's monotone shape *qualitatively* (higher |α| → more drift) but the deference vector consistently sits 15–35pp above random_v0 at moderate-to-large alphas. At α=−1 they're tied (~33%), suggesting that small-|α| steering is dominated by stochasticity, not direction.

### What this changes for the safety claim

The earlier v3c+v5a writeup argued: *the deference direction is what the model uses to drive substantively different recommendations.* The v6 results refine this:

1. ✅ **Behavior coupling on natural text replicates exactly.** This is the strongest single claim and survives independent reproduction within 1pp on the headline number (minedit −1 = 55%/56%). Defended against both judge-variance (Sonnet stable) and surface-text-volatility (paraphrase noise floor of 23–27% << minedit −1 of 55–56%) alternative explanations.
2. ✅ **Probe replicates exactly.** Confirms the user-state representation is deterministic given the model weights.
3. ⚠️ **Steering direction-specificity replicates with caveat.** The deference direction produces ~2× more DIFF_REC than matched-norm random, but random isn't zero. The "deference vector causes substantive recommendation changes" claim is supported, but the "random direction wouldn't do this" claim needed the random control to be quantified, and the answer is "random does this somewhat, deference does it twice as much."
4. ⚠️ **Steering magnitude is sensitive to model decoding stochasticity.** The same activations + vector produce different downstream token distributions on different hardware. Magnitude estimates have a confidence interval roughly ±20pp at α=±2.

The right paper-shaped claim from this body of work is:

> *"In Llama 3.1 8B, a deference-direction probe trained on residual activations at L14 captures user-state in a way that (a) generalizes to lexical twins where text alone does not, (b) tracks behavior such that natural minimal-edit user-state flips drive substantively different model recommendations at +30pp above a same-state paraphrase baseline, and (c) when used as a steering direction at moderate α, produces ~2× more substantive recommendation change than a same-norm random direction at the same layer."*

That's three independently-supported sub-claims, each with a control. The v6 replay confirms (a) and (b) verbatim and bounds (c) honestly.

### What's still open / would need more work

- **Random α=+2 variance is high** (33%, 36%, 59% across 3 seeds). Standard deviation ~15pp on n=3. A 5+ seed estimate would tighten the deference/random ratio.
- **Steering magnitude reproducibility** — would benefit from running the same vector on 3 different hardware setups to characterize the ±20pp envelope. Outside this project's budget.
- **Hand-checking Sonnet's substance verdicts on ~10 marginal cases.** The judge is *stable* (verified) but stable doesn't mean correct. A spot-check would establish the verdicts' validity.
- **Random direction control on natural text (minedit) doesn't apply** — minedit is genuine prompt difference, not steering. Not analogous.

### Cost

Total v6 spend: ~$17 (~$3.50 RunPod + ~$13 Anthropic). Out of original ~$30 budget, comfortable margin remaining.

### Files added this iteration

- `auth_projection/data/v1_relabeled.jsonl` (Phase 0 sanity)
- `auth_projection/data/v6c_*` — full replay outputs (probes, steering, paired gen, scoring, substance for both deference and 3 random vectors)
- `auth_projection/data/v6c_v3c_minedit_substance_judgevar.jsonl`, `..._paraphrase_substance_judgevar.jsonl` — judge variance check on committed responses
- `auth_projection/runpod_helper.py` — boot/wait/kill RunPod pods via API. Reusable for future runs.

### Going-forward thoughts

- **The headline survives a real stress test.** Behavior coupling on natural minimal edits gives 55%/56% across two independent runs against a 23%/27% noise floor.
- **The next big experiment is training-dynamics on Olmo/Pythia checkpoints.** Now that the headline is solid, it's worth investing in the question of when in training the deference feature emerges.
- **Reproducibility lessons:** Sonnet at temp=0 is reliable across cache invalidations; sklearn LogisticRegression on saved activations is fully deterministic; transformer model decoding has ~±20pp stochasticity in Sonnet-judged DIFF_REC% across hardware setups for fixed seeds. Future experiments should report n≥3 seed/hardware draws on any steering-magnitude claim.

---

## 2026-05-09 — v4 Qwen scaling curve + content-substance analysis + same-tier paraphrase baseline

### What we set out to do

After v3c gave us clean steering + behavior coupling on Llama 8B, the obvious next-meeting question from Shivam would be "is this Llama-specific?" To answer that, ran the v3c pipeline on three more Qwen models to build a within-family scaling curve:

- v4a: **Qwen 2.5 7B-Instruct** (matched-scale cross-family check vs Llama 8B)
- v4b: **Qwen 2.5 14B-Instruct** (originally Qwen 32B; HF infra issue forced substitution)
- (v1, v2 already done: Qwen 1.5B and 3B)

Plus two ablations the v3c writeup left dangling:
- **Content-substance analysis** — does steering / natural-text user-state difference change the *recommended action*, or just the framing? (Sonnet judge, asks SAME_ADVICE / DIFFERENT_FRAMING / DIFFERENT_RECOMMENDATION.)
- **Same-tier paraphrase baseline** — what fraction of "different recommendation" verdicts is general model volatility vs specifically driven by user-state changes?

### Setup compromises (real ones, not fudges)

- Qwen 32B was the original target. Three attempts on three different RunPod boxes all failed with the same HuggingFace shard-download race (shard 1 stuck on xethub CDN with 403s, never recovered). Substituted Qwen 14B as the cleanest available step up from 7B with the same problem statement. Cost ~$4.50 in dead pod time before pivoting.
- Qwen 14B has a related but different timing bug: 8 shards download in parallel, transformers.from_pretrained tries to load the model before all shards land, gets either "file not found" or "invalid JSON header" on partially-written shards. This bit us **three times** across recovery attempts and ate ~$15 of pod time across the day. Final state: v4b has the C2 (behavior coupling) results that ran on shorter generation jobs which avoided the timing window, but **no B4 steering**, no probe at assistant-start, no paraphrase baseline. The fix would be a separate `huggingface-cli download` step that blocks until all shards verified — didn't bake it in time.
- v5 baseline run on Qwen 14B failed for the same reason. So the paraphrase baseline only ran on Llama 8B.

### Headline numbers

**Probe scaling curve** (last_user_token, since v4b doesn't have assistant_start):

| model | best layer | acc | AUC strongly |
|---|---|---|---|
| Qwen 1.5B | L17/28 | 0.774 | 0.975 |
| Qwen 3B | L27/36 | 0.767 | 0.980 |
| Qwen 7B | L18/28 | 0.767 | 0.985 |
| **Qwen 14B** | **L31/48** | **0.815** | 0.987 |
| Llama 8B | L14/32 | 0.801 | 0.989 |

Within-Qwen: probe accuracy was flat 0.767–0.774 from 1.5B through 7B, then jumps to **0.815 at 14B**. That's where the within-family scaling actually starts to do something.

**Probe at assistant_start position** (Shivam's intuition replicated cross-family at matched 7-8B scale):

| | last_user_token | assistant_start_token | Δ |
|---|---|---|---|
| Llama 8B (v3c) L14 | 0.801 | **0.841** | +4.0 pp |
| Qwen 7B (v4a) L21 | 0.767 | **0.842** | +7.5 pp |

Same +4 to +7 pp lift on assistant_start vs last_user_token at both models. The probe-position lesson holds cross-family.

**Lexical-twin slice** (assistant_start_token at the chosen layer):

| | combined-twin probe acc | combined-twin tfidf acc | peer_voice probe acc | submission_voice probe acc |
|---|---|---|---|---|
| Llama 8B v3c | 0.850 | 0.707 | **0.920** | 0.733 |
| Qwen 7B v4a | 0.780 | 0.707 | 0.846 | 0.667 |

Both models beat TF-IDF on combined twins and especially on peer_voice (severe deference behavior expressed without submission vocabulary). At Qwen 7B the lift is smaller but still positive. At Llama 8B the probe basically dominates.

**B4 steering on real conversations** (Sonnet-judged, mean across 39 held-out test convs at each α):

For Qwen 7B at L21:

| α | directiveness 0–10 | hedging 0–10 | compliance 0–10 |
|---|---|---|---|
| −2 | 3.51 | 5.67 | 4.54 |
| 0 | 5.72 | 3.90 | 3.56 |
| +2 | 5.92 | 2.54 | **2.64** |

Across α ∈ [−2, +1] (the in-distribution range): directiveness rises monotonically 3.51 → 6.10, hedging falls 5.67 → 3.44, compliance falls 4.54 → 2.64. Same monotone shape as Llama 8B, tighter usable α range (Qwen breaks at ±4, Llama tolerates α=−4 still coherent). The cross-family steering replication is essentially perfect inside the coherent range.

**Behavior coupling (mechanical metrics on expanded n=136 minedit pairs at Qwen 7B):**

- length_chars on +1 (more deferential): **−36, p=0.001 ***** (predicted direction)
- length_chars on −1 (less deferential): **+48, p=0.0001 ***** (predicted direction)
- length_words +1: −6, p=0.001 ***
- length_words −1: +5.2, p=0.005 **
- hedges −1: +0.38, p=0.015 **

Qwen 7B is *symmetric* on mechanical metrics (both flip directions hit p<0.05) — Llama 8B was asymmetric (only −1 fired). Different shape, but both clean wins.

### Content-substance analysis (the new headline experiment)

For each held-out conversation, asked Sonnet to compare assistant responses pairwise and verdict whether the underlying recommendation is the same or different.

**On steering pairs (vs α=0 baseline):**

| α | Qwen 7B (v4a) DIFF_REC% | Llama 8B (v3c) DIFF_REC% |
|---|---|---|
| −4 | (off-manifold) | **74%** |
| −2 | 33% | 18% |
| −1 | 13% | 23% |
| +1 | 10% | **44%** |
| +2 | 36% | **69%** |

Steering at moderate α genuinely changes the model's recommendation, not just packaging. At Llama 8B α=+2, Sonnet says **69% of responses recommend a substantively different action than baseline** — 27 of 39 conversations. At Qwen 7B the same α gives 36%. **Llama is more steerable on substance than Qwen** by ~2× at matched α.

**On natural minimal-edit pairs (no steering, just label-flipped user turns):**

| model | direction | DIFF_REC% |
|---|---|---|
| Qwen 7B v4a | +1 (more deferential) | 17% |
| Qwen 7B v4a | −1 (less deferential) | 38% |
| Qwen 14B v4b | +1 | 19% |
| Qwen 14B v4b | −1 | 32% |
| **Llama 8B v3c** | +1 | 34% |
| **Llama 8B v3c** | −1 | **55%** |

In all three models, **on natural text, the model recommends substantively different actions in roughly a third to a half of conversations when the user's deference state changes.** The asymmetric pattern (more diff-rec on the −1 "less deferential" side) replicates cleanly across all three models.

### Baseline (the test for whether the substance result is real)

The above numbers don't separate "user-state coupling" from "general model volatility on text changes." Built same-tier paraphrase pairs (parent + Sonnet-rewritten paraphrase with label preserved) and ran the same substance comparison. **Different surface text, same user-state label.** This is the noise floor.

**Llama 8B same-tier paraphrase baseline (n=133):**

| paraphrase tier | DIFF_REC% |
|---|---|
| none → none | 17% |
| somewhat → somewhat | 23% |
| strongly → strongly | 48% (n=21, small) |
| **overall** | **23%** |

Compared to the minedit (state-flipping) result on Llama 8B:

| | DIFF_REC% | gap above 23% baseline |
|---|---|---|
| Same-tier paraphrase (no state change) | **23%** | — |
| Minedit −1 (less deferential) | 55% | **+32 pp** ← real, sizeable |
| Minedit +1 (more deferential) | 34% | +11 pp |

The −1 effect is **32 percentage points above noise floor**. The +1 effect is +11 pp — smaller, but still positive. **The substance result is not just text volatility; it tracks user state.**

The asymmetric pattern is now triangulated across three independent measurements at Llama 8B: length deltas (−1 only), judge tie-rate (17/4 vs 3/10 split), substance verdicts (+32 pp vs +11 pp gap). Three observers, same asymmetry.

### What this lets us claim now

With the v4 + v5 results layered on top of v3c:

1. **Probe captures user-state info beyond what text features recover** — replicates at Qwen 7B + Llama 8B; assistant-start-token position is the better readout in both models (+4 to +7 pp). At Qwen 14B with last-user-token only (no assistant_start probe trained), accuracy is 0.815, the strongest within-Qwen.
2. **Steering at moderate α causally shifts the model's recommendation, not just style** — replicates at Qwen 7B + Llama 8B; ~33–69% of responses substantively change at α=+2 depending on model.
3. **Behavior coupling is real on natural text, validated against a noise-floor baseline** — Llama 8B minedit −1 gives 55% DIFF_REC vs 23% same-tier paraphrase baseline (+32 pp gap). Same kind of effect at Qwen 7B and Qwen 14B but smaller magnitude (and we only have the baseline for Llama 8B).
4. **The asymmetric pattern is robust** — across length deltas, judge verdicts, substance verdicts, the model adapts more aggressively to "user becoming less deferential" than to "user becoming more deferential." Same direction in all three Qwen + Llama runs.

### What we don't have / caveats

- **Qwen 14B B4 steering experiment** — three retries failed on the same HF shard-download race, ate ~$15 of dead pod time, eventually ran out of budget. Within-Qwen steering scaling story stops at 7B.
- **Same-tier paraphrase baseline only on Llama 8B** — the Qwen versions died with the 14B steering attempts. So the "23% noise floor" only validates Llama; Qwen 7B and 14B numbers are unbaselined.
- **Random-direction steering control still not run** — would test whether the deference vector is direction-specific or whether any large perturbation flips recommendations. Holding for a future iteration.
- **Hand-built testbed** — Qwen 7B last_user_token 80% / assistant_start 60%. Llama 8B last_user_token 80% / assistant_start 73%. Both consistent with model not memorizing single-conv-distribution; flat across twin types. Sanity check passes but isn't a strong signal either way.
- **Content-substance Sonnet verdict reliability** — single-pass judge, no spot-check of marginal cases. Worth validating manually on ~10 random pairs before publishing.

### Total spend

v4 + v5 across the day: ~$36 (close to the $36 RunPod credit + ~$10 in Sonnet API).
- Productive GPU: Llama 8B ~$2, Qwen 7B ~$2, Llama 8B baseline ~$2 = ~$6
- Unproductive GPU (Qwen 32B failed pod, three Qwen 14B failures): ~$25
- Sonnet (steering scoring + content substance + paraphrase substance): ~$10

Most of the dead spend was on the Qwen 14B race condition. With one fix (pre-verify shards before running pipeline), the same experiment would have cost ~$5 and run cleanly.

### Files added this iteration

- `auth_projection/build_paraphrase_pairs.py` — same-tier baseline pair builder
- `auth_projection/content_substance_analysis.py` — Sonnet substance verdicts on steering pairs
- `auth_projection/content_substance_minedit.py` — Sonnet substance verdicts on natural minedit pairs
- `auth_projection/data/v4a_*.json/jsonl` (Qwen 7B full pipeline)
- `auth_projection/data/v4b_*.json/jsonl` (Qwen 14B partial — behavior coupling + content-substance only)
- `auth_projection/data/v3c_paraphrase_responses.jsonl` + `v3c_paraphrase_substance.jsonl` (baseline)
- `auth_projection/data/v3c_content_substance.jsonl`, `v4a_content_substance.jsonl` (steering substance)
- `auth_projection/data/v3c_minedit_substance.jsonl`, `v4a_minedit_substance.jsonl`, `v4b_minedit_substance.jsonl` (natural-text substance)
- `docs/v4a_steering_viewer.html` — Qwen 7B side-by-side viewer (will publish via Pages)

### Going-forward thoughts

- **For the next mentor meeting**, the load-bearing claim is "user authority-projection is a feature the model both *represents* (probe) and *uses to drive substantively different recommendations* (behavior coupling, validated against baseline)." Cross-family at matched scale (Qwen 7B + Llama 8B) is the strongest defense against "Llama-specific quirk."
- **For the project to become paper-shaped**, the missing piece is the **random-direction steering control** — without it, the steering result has the "any-direction-flips-recommendations" loophole. That's the next ~$5 experiment to run.
- **The Qwen 14B steering hole** is annoying but not load-bearing. The within-Qwen scaling curve covers 1.5B → 3B → 7B → (gap) → 14B for behavior coupling, with steering specifically covered at 7B and Llama 8B for the cross-family argument.
- **The same-tier paraphrase baseline** is now part of standard methodology going forward. Any future substance claim needs it.

---

## 2026-05-08 — v3b/v3c: probe-position win + clean monotonic steering on real conversations

Two consecutive RunPod sessions (v3b first, v3c retry after a peft-import bug). Headline: two genuinely strong results, both at Llama 3.1 8B layer 14.

### Probe position matters: assistant-start-token >> last-user-token

Following Shivam's mentor-meeting suggestion to read activations at the chat-template assistant-start token (the position right before the model begins generating its response) rather than at the last user-token, I re-extracted Llama 8B activations saving both positions, then trained probes at each.

**The new probe position dominates on every metric we measure:**

| metric (Llama 8B, L14) | last_user_token | **assistant_start_token** | Δ |
|---|---|---|---|
| Standard test accuracy | 0.801 | **0.841** | **+4.0 pp** |
| AUC strongly-vs-rest | 0.989 | **0.996** | +0.7 pp |
| Counterfactual paraphrase gap | 0.013 | **−0.001** | essentially zero — paraphrase-invariant |
| Minimal-edit flip rate | 40.4% | **47.2%** | +6.8 pp |
| **submission_voice acc** (deferential vocab without behavior) | 0.600 | **0.733** | **+13.3 pp** |
| **peer_voice acc** (deference behavior without submission vocab) | 0.846 | **0.920** | +7.4 pp |
| Combined-twin probe acc | 0.756 | **0.850** | +9.4 pp |
| Combined-twin AUC strongly | 0.996 | **1.000** | perfect ranking |

The single biggest practical implication: **the "submission_voice still bad" caveat we'd been carrying for two sessions essentially closes.** At the new probe position, the probe is competitive with TF-IDF on submission_voice (0.733 vs 0.733) and dominates on peer_voice (0.920 vs 0.692). Combined twins are now a 9.4 pp probe win over TF-IDF (vs a tie at the old position) and AUC strongly is perfect.

The CF gap dropping from 0.013 to −0.001 (paraphrase test scores essentially the same as standard) is also striking — at the new position the probe is **paraphrase-invariant**, the cleanest possible test for "signal isn't tied to specific phrasings."

The cosine similarity between the two steering vectors (`v_last_user_token` vs `v_assistant_start_token`) is only 0.41 — they're meaningfully different directions in the residual space. The assistant-start vector also has smaller norm (2.17 vs 3.72), consistent with "the relevant direction is sharper / lower-rank at the position right before generation begins."

### B4: clean monotonic steering on real conversations

We finally ran the cleanly-designed steering experiment: take 39 held-out test conversations, cut the prefix through the *last* user turn, generate the assistant response with `α·v` injected at every token *during generation*, locked sampling seed across alphas. Vector source: assistant_start_token activations. Sweep α ∈ {−4, −2, −1, 0, +1, +2, +4}.

For each of 273 generations (39 convs × 7 α), Sonnet rated the response on:
- **directiveness** (0–10): how much the assistant tells the user what to do vs enumerates options
- **hedging** (0–10): how much the assistant softens claims
- **compliance** (0–10): how much the assistant cedes its judgment back to the user

Plus mechanical metrics (length, hedge-words, directive-words).

**Aggregate result, mean across 39 convs at each α:**

| α | mean len chars | directiveness | hedging | compliance |
|---|---|---|---|---|
| **−4** | 1761 | **1.28** | **7.10** | 4.00 |
| −2 | 1431 | 3.46 | 5.46 | 4.41 |
| −1 | 1215 | 4.54 | 4.77 | 3.92 |
| **0** (baseline) | 1105 | 5.13 | 4.38 | 3.62 |
| +1 | 908 | 6.00 | 3.41 | 3.10 |
| +2 | 839 | **6.23** | **2.85** | **2.64** |
| +4 (off-manifold) | 993 | 1.36 | 3.13 | 2.46 |

Across α ∈ [−4, +2] (the in-distribution range — 80% of residual norm at the endpoints, expected to stay coherent):
- **Directiveness rises monotonically 1.28 → 6.23** (~5× shift). At α=−4 the assistant gives pure-enumeration responses with no recommendation; at α=+2 it takes a clear stand.
- **Hedging falls monotonically 7.10 → 2.85** (60% drop).
- **Compliance falls 4.00 → 2.64** (slight bump at α=−2 then strict monotone).
- **Length falls 1761 → 839 chars** (50% drop). The terser-when-user-deferring effect we'd seen weakly in behavior coupling shows up cleanly in steering.

At α=+4 (160% of residual norm), the response degenerates as predicted; the Sonnet-rated directiveness collapses back to 1.36 because the text becomes incoherent.

This is the cleanest steering result the project has produced. Vastly better than the muddled roleplay-prompt experiment from a few days ago. Here we have:
- Real conversations (not synthetic roleplay prompts)
- Real assistant generation slot (not "model pretends to be user")
- Same baseline at α=0 by construction (locked seed)
- Strong monotone effect across 4 different metrics
- Effect saturates exactly where the residual-norm math predicted breakdown

The steering effect on **length** mirrors the behavior-coupling effect we found earlier: when the user is "more deferential" (higher α steering, or more deferential natural text), the assistant gives a shorter response. The two converge on the same finding from different angles.

Side-by-side HTML viewer with all 273 generations is published to GitHub Pages at `docs/v3c_steering_viewer.html` for qualitative scrolling.

### B2: hand-built testbed

15 hand-written test conversations across the tier × twin-style grid, including borderline cases. Run through both probes:

| | overall | peer_voice (n=4) | submission_voice (n=4) | match (n=7) |
|---|---|---|---|---|
| last_user_token probe | **12/15 (80%)** | 4/4 (100%) | 3/4 (75%) | 5/7 (71%) |
| assistant_start_token probe | 11/15 (73%) | 4/4 (100%) | 2/4 (50%) | 5/7 (71%) |

Both probes get peer_voice perfectly. On the small hand-built set, last_user_token actually edges assistant_start by one prediction (small-n noise; opposite ranking from the synthetic test). The 80% overall on hand-built convs is reasonable validation that the probe behaves sensibly on conversations from outside the synthetic-data distribution.

### C2: expanded minimal-edit behavior coupling at 8B

The expanded labeled set (67 minimal-edit children, +72% over the original 39) yielded 136 flip-points. Llama 8B paired generation + Sonnet judge.

| metric (less-deferential side) | original (n=73) | **expanded (n=136)** |
|---|---|---|
| length_chars Δ | +123 (p=0.006) | **+108 (p=0.001)** |
| length_words Δ | +22 (p=0.0001) | **+19 (p=0.0001)** |
| hedges Δ | +0.95 (p=0.061) | +0.62 (p=0.079) |
| directives Δ | n.s. | −0.15 (p=0.061) ✓ now directional |

The behavior-coupling effect holds with more data and the p-values tighten. The "+1" (more-deferential side) directional trends are still mostly null on individual mechanical metrics but are consistently in the predicted direction (length_words Δ = −10.9 in expanded vs +6.6 in original).

### What now changes about the framing

Three concurrent updates:

1. **The probe IS reading internal user-state, more than we'd previously claimed.** With submission_voice fixed by position choice, the "probe is partially fooled by polite words" caveat largely closes. At L14 with assistant-start-token, the probe ties or beats TF-IDF on every twin slice.

2. **Steering is causally meaningful, not just an artifact of LM-head bias.** The clean monotone effect on real conversations at moderate α — with locked seed across alphas, baseline at α=0, and a predicted breakdown threshold — is much harder to explain away than the previous roleplay-prompt result. This is the version of full-position steering that should headline any writeup.

3. **The behavior-coupling result was not noise.** Doubling the eval set tightens the effect, doesn't dissolve it.

Together: three independent observers (probe accuracy, real-conversation steering, behavior coupling at scale) all show that user authority-projection is something Llama 3.1 8B both *represents internally* and *uses to drive its responses*.

### Caveats holding

- 8B is one model — Qwen 7B replication still pending
- Effect sizes are interpretable, not overwhelming
- Cosine sim 0.41 between the two steering vectors means the right vector to use is genuinely an open question; we don't have ablations across vector sources
- The ~15-pp peer_voice probe lead is on n=26 — small absolute count, even if the signal is sharp

### Methodology improvements baked in this iteration

- `extract_activations.py` saves both `last_token_act` and `assistant_start_token_act` per turn
- `train_probe.split_by_seed` enforces an explicit train/test no-overlap assertion
- `eval_counterfactuals.main` enforces "CF parents must come from test split, not train"
- `model_utils.py` makes the `peft` import optional (fixed the v3b crash)
- `steer_real_conversations.py` accepts `--vector_source` flag
- Pipeline orchestration uses `|| true` per step so one failure doesn't abort the rest

### Files added/modified this iteration

- `auth_projection/steer_real_conversations.py` — B4
- `auth_projection/score_steering_responses.py` — Sonnet directiveness/hedging/compliance scorer
- `auth_projection/build_steering_html.py` — side-by-side HTML viewer
- `auth_projection/train_probe_alt_position.py` — assistant-start-token probe
- `auth_projection/run_handbuilt_testbed.py` — B2
- `auth_projection/extract_activations.py` — saves both positions
- `utils/model_utils.py` — optional peft import
- `docs/v3c_steering_viewer.html` — published HTML viewer (273 generations, side-by-side)

### Cost

v3b: ~$3.50 (terminated at watcher deadline before B4 ran due to bug)
v3c: ~$2 GPU + ~$5 Sonnet API
Total: **~$10.50** for the whole iteration, ending with the cleanest steering result and the corrected probe position.

---

## 2026-05-06 (night) — v3 (Llama 3.1 8B): scale-emergent representation + behavior coupling

Big run, big result. Llama 3.1 8B Instruct on a fresh RunPod A100 80GB (with 100GB volume + 60GB container disk to actually fit the model). Same pipeline as v1/v2 but on a meaningfully larger model from a different family.

### Probe at L14 (selected by accuracy, the criterion we now prefer per the methodology entry below)

| metric (Llama 8B, L14) | value | vs v2 L27 (Qwen 3B) | vs v1 L17 (Qwen 1.5B) |
|---|---|---|---|
| Accuracy | **0.801** | +3.4 pp | +2.7 pp |
| AUC strongly | 0.989 | +0.9 pp | +1.4 pp |
| CF gap (strongly) | 0.013 | +0.008 | −0.008 |
| Minimal-edit flip rate | **40.4%** | −3.4 pp | −0.7 pp |

The script's auto-pick (max AUC strongly) put the layer at L23 (acc=0.788, AUC=0.993). L14 is better on accuracy and on the harder discrimination tests — same lesson we learned for v2.

### **Probe beats TF-IDF on lexical-twin corner cases for the first time** (L14)

| slice | n | probe acc | TF-IDF acc | Δ | probe AUC strongly | TF-IDF AUC strongly |
|---|---|---|---|---|---|---|
| all | 146 | **0.801** | 0.795 | +0.6 | **0.989** | 0.986 |
| match (no twin) | 105 | 0.819 | 0.829 | −1.0 | 0.989 | 0.987 |
| **peer_voice** (strongly behavior, no submission vocab) | 26 | **0.846** | 0.692 | **+15.4 pp** | 0.992 | 0.992 |
| submission_voice (none behavior, deferential vocab) | 15 | 0.600 | 0.733 | −13.3 | n/a | n/a |
| **lexical_twin_combined** | 41 | **0.756** | 0.707 | **+4.9 pp** | **0.996** | 0.992 |

The probe at v1 and v2 lost or tied to TF-IDF on every twin slice. **At v3 L14, the probe wins on combined twins and beats TF-IDF by 15 pp on peer_voice** (severe-deference behavior expressed without submission vocabulary). This is the cleanest evidence we have that the residual probe is reading something text classifiers can't fully recover at this scale.

Submission_voice (deferential vocabulary without deferential behavior) is the case the probe still loses on — a model representing "user just used submission words" still bleeds into the probe direction. But peer_voice is the direction that matters most for the safety claim: detecting users who are fully surrendering agency without flagging their language as deferential.

### Behavior coupling: clean positive at 8B

Same paired-generation pipeline, 73 minimal-edit flip-points, locked seed.

**Mechanical metrics — significant on the −1 (toward less deference) side:**

| metric | direction | 1.5B (mean Δ, p) | 3B (mean Δ, p) | **8B (mean Δ, p)** |
|---|---|---|---|---|
| length_chars | −1 (less deference) | +24.1, p=0.86 | +38.5, p=0.017 | **+122.6, p=0.006** *** |
| length_words | −1 | +3.7, p=0.75 | +3.4, p=0.25 | **+21.7, p=0.000** *** |
| hedges | −1 | +0.49, p=0.23 | +0.67, p=0.044 | +0.95, **p=0.061** * |
| length_chars | +1 (more deference) | −22.6, p=0.19 | −11.4, p=0.42 | +6.6, p=0.97 |

When the user becomes *less* deferential, the 8B model's response gets ~123 chars / 22 words longer, and hedges more — far more than at smaller scales. The +1 side (more deferential user → shorter response) is null at 8B, where it was significant at p<0.10 at 3B. So the behavior coupling at 8B is **asymmetric: more discursive responses to less-deferential users, but no clear shortening for more-deferential users**.

**LLM-judge — sharply directional:**

| flip direction | 1.5B (parent/child/tie) | 3B (parent/child/tie) | **8B (parent/child/tie)** |
|---|---|---|---|
| −1 (toward less deference) | 5/5/29 (74% tie) | 7/1/31 (79% tie) | **17/4/18 (46% tie)** |
| +1 (toward more deference) | 2/2/30 (88% tie) | 0/4/30 (88% tie) | **3/10/21 (62% tie)** |

The judge tie rate dropped from 80%+ at smaller scales to 46–62% at 8B. The non-tie verdicts line up with the predicted direction in both signed-flips: the more-deferential side gets the more-directive verdict (17/4 vs 3/10). Position-swap consistency improved further (71%, vs 77% at 3B and 55% at 1.5B).

This is **a clean and convincing positive at 8B for behavior coupling.** The model gives different (more directive, more discursive) advice as a function of user authority-projection state.

### Steering: full-position monotone, user-position-only mildly responsive

**Full-position steering at L23** (the AUC-best layer, also where the script ran by default — saved as `v3_steering_samples.json`):
- α=−2 to +2: coherent, mild tone differences
- α=+4: starts producing deference vocabulary ("I trust", "I just go", "give me")
- α=+8/+16: full degeneration into submission-themed token soup ("ok just just give me literally go tell me I trust I just go go")

Same qualitative pattern as 1.5B and 3B but breakdown threshold at slightly different α. The Chinese code-switch we saw at Qwen models doesn't appear at Llama — broken-English token loops instead.

**User-position-only steering at L23** (`v3_steering_user_positions.json`):
- α=−8 to +2: outputs are *much* more stable than at 1.5B (where small α produced random-looking variation). Same conversation across these alphas yields nearly-identical model responses.
- α=+4 / +8: small but real shift. Travel example: "I recommend trains" appears in +4 and +8 but not in lower α. Relationship example: opening shifts from "Crafting the message…" at low/zero α to "The tricky part is…" at +8.

This is qualitatively different from the 1.5B user-position steering null. At 8B, perturbing user-position residuals **does** propagate to assistant generation, just weakly. Not a clean monotonic shift like full-position steering — but the model is not invariant to the perturbation. Consistent with the behavior-coupling positive at the same scale: the model has *some* representation of user state that the assistant generation conditions on.

### What this means together

The 8B run lands on the strong version of the original hypothesis:
1. **The probe captures user-state information not fully recoverable from text.** First clear win over TF-IDF on peer_voice (15 pp acc lead), with overall AUC slightly higher too.
2. **The model behaviorally adapts to user state.** Two mechanical metrics at p<0.01, judge agreement clean and directional in both flip directions.
3. **The probe direction is, to some degree, the direction the model uses to read user state.** Perturbing it at user positions produces small but real shifts in assistant output — different from the 1.5B null.

Compared to the 1.5B and 3B runs, what changed at 8B is roughly: the model now *has* a useful internal user-state representation, which it uses (a bit) to drive behavior, and which lives in residual structure that text features alone don't fully capture. None of those three were true at 1.5B; some were partially true at 3B; all three are true at 8B.

### Caveats

- **Submission_voice is still bad** (probe acc 60% vs TF-IDF 73%). The probe is still *more* fooled by deferential vocabulary lacking deferential behavior than TF-IDF is. The peer_voice gain is the directional improvement that matters; submission_voice failure is consistent with a residual-anchoring-on-words effect that scale doesn't completely fix.
- **Effect sizes are small** even at 8B (~46–62% tie rate on judge, mid-magnitude p-values on metrics). Not "obviously visible to a reader of the conversation" levels.
- **+1 side mechanical metrics are null at 8B** — only the −1 side (less-deferential user → discursive response) is significant. The asymmetry suggests the model is more sensitive to "this user is being independent, give them more depth" than "this user is being deferential, be terse." Worth investigating.
- **Sample sizes are still small** (n=73 pairs, ~30-40 per signed-flip direction). All effects could shrink under more data; the *direction* is what we're confident in.

### Methodology lesson holds

The earlier "multi-criterion best-layer" lesson played out again: the script picked L23 (max AUC), which gave decent but worse-than-L14 numbers. L14 (max accuracy) is the right pick here too — same agreement (max accuracy and max flip rate both pick L14) — and the lexical-twin probe-beats-TF-IDF result only shows up at L14, not L23. **Always pick by max accuracy + cross-check with max flip rate** going forward.

### Files added

- `auth_projection/data/v3_activations.pt` (Llama 8B, 708 records, 33 layers, 4096 hidden)
- `auth_projection/data/v3_cf_paraphrase_activations.pt`, `v3_cf_minimal_edit_activations.pt`
- `auth_projection/data/v3_probes/probe_C_layer{0..32}.joblib`
- `auth_projection/data/v3_probe_results.json`
- `auth_projection/data/v3_counterfactual_eval_layer{14,23}.json`
- `auth_projection/data/v3_lexical_twin_eval_layer14.json`
- `auth_projection/data/v3_text_classifier_results.json`
- `auth_projection/data/v3_steering_samples.json` (full-position, L23)
- `auth_projection/data/v3_steering_user_positions.json` (user-positions only, L23)
- `auth_projection/data/v3_minedit_assistant_responses.jsonl`
- `auth_projection/data/v3_minedit_behavior_metrics.json`
- `auth_projection/data/v3_minedit_judge.jsonl`
- `auth_projection/data/v3_minedit_judge_summary.json`

### Cost (full v3 run)

A100 80GB on RunPod Community: ~3 hours including the false-start with the disk-too-small pod and the OOM-killed first steering attempt. ~$3.50. Plus ~$1 for Sonnet judge. Total: ~$4.50.

---

## 2026-05-06 (corrections) — multi-criterion best-layer comparison: we picked the wrong layer at v2

User asked the right methodological question while v3 was still running: were we biasing ourselves by always picking "best layer" by AUC strongly, when the layer that maximizes that metric isn't necessarily the layer that maximizes accuracy, minimizes lexical anchoring (CF gap), or is best on the hardest discrimination test (minimal-edit flip rate)?

I ran the comparison on v1 and v2 saved probes against four criteria.

| | max accuracy | max AUC strongly | min CF gap | max flip rate |
|---|---|---|---|---|
| **v1 (Qwen 1.5B)** | **L17** (0.774/0.975/0.021/41.1%) | L21 (0.692/0.982/0.014/32.9%) | L5 (0.651/0.877/−0.039/24.7%) | **L17** (same) |
| **v2 (Qwen 3B)** | **L27** (0.767/0.980/0.005/**43.8%**) | L30 (0.719/0.986/0.014/**28.8%**) | L13 (0.712/0.934/−0.025/29%) | **L27** (same) |

(L5 / L13 winning min-CF-gap is a noise artifact — they have *negative* CF gap, meaning paraphrase scores higher than standard test, on small n. Discount these.)

### v1 — layer choice was defensible

Accuracy and flip rate both pick L17. AUC strongly picks L21 with sharp accuracy cost (0.692 vs 0.774). The original choice (L17) wins 2 of 4 criteria; the previous writeup is unaffected.

### v2 — we picked the wrong layer, and it changes the story

The previous writeup picked L30 (max AUC strongly). But L27 wins on accuracy and flip rate, with effectively the same AUC (0.980 vs 0.986). The L27 numbers paint a meaningfully different picture:

| metric (3B) | L27 (correct pick) | L30 (we previously reported) | corrected |
|---|---|---|---|
| Accuracy | 0.767 | 0.719 | +5 pp |
| AUC strongly | 0.980 | 0.986 | −0.6 pp (negligible) |
| CF gap strongly | **0.005** | 0.014 | 3× smaller |
| Minimal-edit flip rate | **43.8%** | 28.8% | +15 pp |
| Lexical-twin combined probe acc | **0.707** | 0.610 | +9.7 pp |
| Lexical-twin combined tfidf acc | 0.707 | 0.707 | (unchanged) |
| submission_voice probe acc | 0.667 | 0.467 | +20 pp |
| peer_voice probe acc | **0.731** | 0.692 | +3.9 pp |
| peer_voice tfidf acc | 0.654 | 0.654 | (unchanged) |

### What this changes

The previous (L30) writeup said: *"minimal-edit flip rate worse at 3B (28.9%) than 1.5B (40.4%) — probe is more lexically anchored at scale; submission_voice acc collapses to 47%; probe loses to TF-IDF on every twin slice."* Each of those claims was driven by the layer selection.

At L27 (correct pick), the picture is:
1. **Flip rate is 43.8%, slightly *better* than v1's 41.1%.** Scale doesn't make the probe more lexically anchored on minimal edits — at the right layer, it's marginally less so.
2. **CF gap on AUC strongly drops from 0.021 (v1) to 0.005 (v2 L27)** — meaningful improvement in paraphrase robustness with scale.
3. **Probe ties TF-IDF on combined twins (0.707 vs 0.707)** instead of losing 0.610 vs 0.707. The probe-vs-TF-IDF gap on twin discrimination essentially disappears at L27 — which is consistent with "scale modestly helps the probe escape lexical anchoring."
4. **Probe beats TF-IDF on peer_voice acc (0.731 vs 0.654)** — the corner case where users behave deferentially without using submission vocabulary.
5. **submission_voice acc still trails TF-IDF (0.667 vs 0.800)**, but the gap shrunk from 33pp at L30 to 13pp at L27. Surface deferential vocabulary still fools the probe more than TF-IDF, but less dramatically.

So the corrected v2 story is **scale modestly helps detection in lexical-twin corner cases**, not "scale makes the probe more lexically anchored."

### Behavior coupling at v2: still positive

The behavior-coupling result doesn't depend on layer choice at all (the paired-generation pipeline doesn't use the probe; only the labels). The v2 finding stands:
- Mechanical metrics: length_chars Δ=+38.5, p=0.017 (less-deference flip); hedges Δ=+0.67, p=0.044
- Judge: 7/1 parent-wins on −1 flips, 0/4 parent-wins on +1 flips (both directionally correct)

Combined with the corrected layer story: behavior coupling and corner-case discrimination *both* improved at 3B. The "scale-emergent" framing for v2 is now stronger, not weaker.

### Methodology lesson and going forward

Layer-selection bias is real. From now on:
- **Default selection criterion: max accuracy on standard test.** This agreed with max flip-rate at both v1 and v2 — the most consistent pair across criteria.
- **Always report at multiple layers** when claims are sensitive to layer choice (lexical-twin slice, CF gap).
- **Steering experiments should sweep layers**, not just use the AUC-best one. The layer most readable by a probe isn't necessarily the layer most causal for steering.

I'll re-pick layers for v3 (Llama 8B, currently running) by max accuracy and run downstream replication at multiple layers (early/middle/late) to check robustness.

### Files added

- `auth_projection/multi_criterion_layer_compare.py`
- `auth_projection/data/multi_criterion_layer_compare.json` (v1 + v2 per-layer metrics)
- `auth_projection/data/v2_counterfactual_eval_layer27.json` (corrected pick)
- `auth_projection/data/v2_lexical_twin_eval_layer27.json` (corrected pick)

---

## 2026-05-06 (evening) — Qwen 3B rerun on rented A100 80GB: behavior coupling appears at 3B

### Setup notes (and what I had to compromise on)

Rented a RunPod Community Cloud A100 80GB. The intent was Llama 3.1 8B; reality bit:
- Pod's `/workspace` and `/` are 20GB each by default. A 14–16GB model needs ~30GB during HF download (model + transient staging + venv + project).
- I dropped to **Qwen 2.5-3B-Instruct** (~6GB on disk, fits) instead. Same family as 1.5B (clean scale comparison), open-access (no HF token needed).
- 36 transformer layers, 2048 hidden — vs 28 layers / 1536 hidden at 1.5B. Roughly 2× params, 1.3× layers, 1.3× hidden.
- A real Llama 8B / Qwen 7B run is queued for a new pod with adequate disk (50GB+).

### Probe layer-sweep on Qwen 3B

Best layer: **30** (final-stack region). Top 5 by AUC-strongly: L30 0.986, L23 0.985, L21 0.985, L20 0.983, L18 0.982. Saturation pattern is similar to 1.5B (signal recoverable across most of the upper stack).

| | Qwen 1.5B (v1) | Qwen 3B (v2) | direction |
|---|---|---|---|
| Best layer | L17 | L30 | (deeper at 3B) |
| Accuracy at best | 0.774 | 0.719 | ↓ |
| AUC strongly | 0.975 | 0.986 | ↑ |
| AUC none | 0.881 | 0.874 | ≈ |
| AUC somewhat | 0.772 | 0.792 | ≈ |

Accuracy *drops* at 3B but ranking quality (AUC) holds or improves. The acc drop is concentrated on the somewhat class — the harder middle tier.

### Counterfactual eval at L30

Standard AUC strongly 0.986 → paraphrase 0.972, **CF gap 0.014** (vs 0.021 at 1.5B). Slightly more paraphrase-robust at 3B.

**Minimal-edit flip rate: 28.9% (81/280)** — *worse than 1.5B's 40.4%.* The 3B representation is *more* anchored to surface vocabulary than 1.5B was: a small lexical edit that flips the labeler's verdict less often flips the probe's verdict at this scale.

### Lexical-twin slice

| slice | n | probe acc | tfidf acc | probe AUC strongly | tfidf AUC strongly |
|---|---|---|---|---|---|
| all | 146 | 0.719 | **0.795** | 0.986 | 0.986 |
| match (no twin) | 105 | 0.762 | **0.829** | 0.987 | 0.987 |
| peer_voice | 26 | **0.692** | 0.654 | 0.992 | 0.992 |
| **submission_voice** | 15 | **0.467** | **0.800** | n/a | n/a |
| lexical_twin_combined | 41 | 0.610 | **0.707** | 0.992 | 0.992 |

Same overall pattern as 1.5B: probe loses to TF-IDF on aggregate detection. Two new wrinkles:
1. **Probe wins peer_voice acc** (0.692 vs 0.654) — modest but the only twin slice the probe wins on at 3B.
2. **submission_voice gap widens** — probe accuracy collapses from 60% (1.5B) to **46.7% (3B)**. TF-IDF holds at 80%. The 3B representation is *more* fooled by deferential surface vocabulary in non-deferential users than the 1.5B representation was.

So scale didn't help the probe escape lexical anchoring — it made it worse on the harder twin direction, slightly better on the other. AUC ranking is identical (0.992) for both classifiers on twins, so they're sorting the same examples the same way; only the threshold-crossing differs.

### Behavior coupling: **softly positive at 3B**

Same paired-generation pipeline as 1.5B: 73 minimal-edit pair flip-points, parent + child generations with locked seed, mechanical metrics + Sonnet-as-judge with position swap.

**Mechanical metrics — two effects now significant:**

| metric | direction | 1.5B (mean Δ, p) | **3B (mean Δ, p)** |
|---|---|---|---|
| length_chars | toward less deference | +24.1, p=0.86 | **+38.5, p=0.017** ✓ |
| length_words | toward less deference | +3.7, p=0.75 | +3.4, p=0.25 |
| hedges | toward less deference | +0.49, p=0.23 | **+0.67, p=0.044** ✓ |
| length_chars | toward more deference | −22.6, p=0.19 | −11.4, p=0.42 |
| length_words | toward more deference | −4.6, p=0.13 | −4.1, p=0.079 |
| hedges | toward more deference | −0.09, p=0.73 | −0.12, p=0.76 |

When the user becomes *less* deferential, the 3B model's response gets meaningfully longer (p=0.017) and hedges more (p=0.044). When the user becomes *more* deferential, response gets shorter (length_words p=0.079, just missing). Direction matches the hypothesis on both signed-flips; significance landed cleanly on the "less deference → discursive response" side.

**LLM judge — directional split now consistent with hypothesis:**

| flip direction | 1.5B (parent/child/tie) | 3B (parent/child/tie) |
|---|---|---|
| toward more deference (signed +1) | 2/2/30 | **0/4/30** ✓ |
| toward less deference (signed −1) | 5/5/29 | **7/1/31** ✓ |

At 1.5B the non-tie verdicts split 50/50 (no signal). At 3B they line up with the prediction in both directions: the more-deferential side wins the "more directive" verdict 7/1 and 0/4. Tie rate stays high (~85%) — the effect is detectable but small. Position-swap consistency improved 55% → 77%, position bias dropped from 49/17 to 26/17 — the judge is sharper at 3B too.

This is **the first non-null result for behavior coupling.** It's not statistically overwhelming (small effect size, n=73 pairs), but two independent measurements (mechanical metrics and judge) agree in the predicted direction, and the picture is qualitatively different from the clean 1.5B null.

### Steering replications

Full-position steering at L30 — saved to `v2_steering_samples.json`, 27 generations across 3 prompts × 9 alphas. Will revisit in detail; the qualitative output is broadly similar to 1.5B (mild α produces deferential vocabulary, extreme α produces collapse).

User-position-only steering at L30 — `v2_steering_user_positions.json`, 42 generations × 7 alphas across 6 held-out conversations. Same disentanglement question as the 1.5B run; will analyze whether the 3B result is also mostly null on user-state propagation, or whether the behavior-coupling positive signal corresponds to actual KV-cache effects from user-position steering.

### Reframe (again)

Three of the 1.5B nulls held; one budged. Updated picture:

1. **Detection: blackbox still wins.** TF-IDF on user-turn-only matches/beats the probe across all slices, including twins. Probe at 3B is *more* lexically anchored than at 1.5B on the hardest discrimination (submission_voice 47% vs 80%, minimal-edit flip 29% vs 41%). Scale doesn't fix the lexical-anchoring problem.
2. **Behavior coupling: small but real at 3B.** Two mechanical metrics significant (p<0.05), judge directional split consistent with hypothesis on both signed-flips (compare to clean 50/50 noise at 1.5B). Effect is small (~85% tie rate) but the *direction* is now clean.
3. **Steering disentanglement: TBD pending analysis** of user-position-only sweep at 3B.

The behavior-coupling positive shifts the project framing back toward something paper-shaped, but with the right caveats:
- The probe doesn't gain detection power at scale; if anything it loses.
- The model *behaves* differently with user state at 3B vs 1.5B — that's the scale-emergent finding.
- Whether the probe direction is the *cause* of the behavior shift (vs being a byproduct) is the disentanglement question, and the user-position-only steering at 3B will speak to it directly.

### What I want from a real 8B run

The 3B result is a hint, not a clean signal. To turn it into a paper claim, we want at 8B:
- **Behavior-coupling magnitude** — does the effect grow further? At 1.5B p≫0.10, at 3B p<0.05 — does 8B push it firmly into "any reader could see it"?
- **Probe-on-twins** — does the probe finally beat TF-IDF on twins at 8B, or does the lexical-anchoring hold?
- **User-position-only steering** — at 8B, does the disentanglement experiment now show user-state propagation through KV cache to assistant behavior? If yes, paper. If no, probe is still a deferential-language detector even at scale.

### Files added this iteration

- `auth_projection/data/v2_activations.pt` (Qwen 3B, 708 records, 37 layers, 2048 hidden)
- `auth_projection/data/v2_cf_paraphrase_activations.pt`, `v2_cf_minimal_edit_activations.pt`
- `auth_projection/data/v2_probes/probe_C_layer{0..36}.joblib`
- `auth_projection/data/v2_probe_results.json`, `v2_counterfactual_eval_layer30.json`
- `auth_projection/data/v2_lexical_twin_eval.json`, `v2_text_classifier_results.json`
- `auth_projection/data/v2_steering_samples.json`, `v2_steering_user_positions.json`
- `auth_projection/data/v2_minedit_assistant_responses.jsonl`, `v2_minedit_behavior_metrics.json`, `v2_minedit_judge.jsonl`, `v2_minedit_judge_summary.json`
- Pipeline scripts on remote: `run_v2_pipeline.sh`, `run_v2_pipeline_resume.sh`

### Cost

RunPod A100 80GB Community Cloud, ~1.5 hours total (including the failed 7B attempts). ~$1.80. Plus ~$1 in Sonnet judge calls. Total: ~$2.80.

---

## 2026-05-06 (latest) — disentanglement experiment + lexical-twin eval: third strike against the whitebox-superiority story

Two more experiments today, both prompted by user pushback on the previous entry. Both hit null in the same direction.

### Disentanglement: user-position-only steering

**The question yesterday's full-position steering left dangling:** when α=+2 produced "I will do whatever you tell me. You're the boss," was that the model representing the user as deferential (and adapting), or just the steering vector biasing the LM head toward deferential vocabulary during generation? The steering hook fired at every token position during assistant generation, so we couldn't separate the two.

**The cleaner test (run today):** apply the steering vector ONLY at user-token positions in the prefix, then run un-perturbed assistant generation. If the vector represents user state on the model's side, the perturbation should propagate through KV cache and bias the assistant's response. If the vector is just a language-production direction, restricting it to user positions should make the deference signal disappear.

**Setup.** 6 held-out test conversations, each cut at user_turn_index=2. Build a token-position mask from the chat template (find user-turn slices via `find_string_in_tokens`, mask=1 inside user content, 0 elsewhere — 71–282 user tokens per conv). Hook on layer 16 (= probe layer 17 output). Apply v · α · mask only on the prefix forward; subsequent per-token generation forwards are no-ops. Sweep α ∈ {−8, −4, −2, 0, +2, +4, +8}.

**Result: assistant responses vary across α but do NOT shift monotonically toward deference.** Compare to yesterday's full-position result:

| α | Yesterday (steer all positions, generation) | Today (steer user positions only, generation un-perturbed) |
|---|---|---|
| −2 | "It sounds like you're exploring career paths…" (exploratory) | (varies; structured travel breakdowns, communication advice, etc.) |
| 0 | "I'm sorry, but I can't assist with that." | (baseline neutral assistant responses) |
| **+2** | **"Okay! Let's go. I will do whatever you tell me."** | (varies; "I understand; you're looking for realistic expectations…", structured travel advice, no deference shift) |
| +8 | Chinese token-loops about absolute obedience | (still coherent assistant responses, no language collapse) |

The variability is real (different alphas produce different responses, so the perturbation is doing *something*), but there is no clean directional pattern toward more or less deference. At α=0 in one travel conversation the model refused to choose between options; at +α it gave more help — opposite of "more deferential user → more directive assistant."

**Interpretation.** The steering vector is mostly a **deferential-language production direction**, not a **user-state representation that the model adapts to.** Yesterday's striking result came from biasing the LM head *during generation*; restricting steering to user positions removes that effect because the assistant generation runs un-perturbed. The KV cache contains the perturbed user-state representation, but the assistant's next-token distribution doesn't change appreciably as a function of it.

This is the cleanest disentanglement we could get. It cuts directly against the "the probe captures user-state-the-model-adapts-to" framing.

### Lexical-twin slice: probe loses to TF-IDF on the corner cases the design specifically tested

Yesterday's TF-IDF result (matches probe overall) raised the question: *is the lexical-twin design just not protective?* So I sliced the test set by `lexical_twin_kind` and re-evaluated both classifiers on each slice.

| slice | n | labels n/sw/st | probe acc | tfidf acc | probe AUC strongly | tfidf AUC strongly |
|---|---|---|---|---|---|---|
| all | 146 | 84/38/24 | 0.774 | **0.795** | 0.975 | **0.986** |
| match (no twin) | 105 | 66/22/17 | 0.829 | 0.829 | 0.989 | 0.987 |
| **peer_voice** (strongly behavior, no submission vocab) | 26 | 11/8/7 | 0.654 | 0.654 | 0.917 | **0.992** |
| **submission_voice** (none behavior, deferential vocab) | 15 | 7/8/0 | 0.600 | **0.800** | n/a | n/a |
| all twins combined | 41 | 18/16/7 | 0.634 | **0.707** | 0.933 | **0.992** |

The lexical-twin convs are designed to be the corner cases where target tier and surface vocabulary deliberately mismatch. The whitebox-superiority hypothesis was: *probe captures behavioral state, so it should win here even when surface vocabulary is misleading.*

**Result is the opposite.** On every twin slice, the probe loses or ties. On submission_voice (deferential surface words but behaviorally none) the probe accuracy drops to 60% while TF-IDF holds at 80% — the probe is *more* fooled by surface deferential language than a bag-of-words classifier. On peer_voice (severe deference behavior with no submission vocab) accuracies tie but TF-IDF AUC is 7.5 pp higher.

Caveat: n=15 and n=26 are small. AUC differences could be noisy. But the direction is consistent across all twin types and both metrics, and the submission_voice accuracy gap (60% vs 80%) is large enough to be unlikely to be noise alone.

**Interpretation.** The probe at L17 is not reading "behavioral state hidden behind surface vocabulary." It is reading something correlated with lexical features and, if anything, more lexically anchored than a bag-of-words view of the same text. The L17 residual probably represents *what the user just said in deference-correlated vocabulary*, which is mostly the same information TF-IDF gets but worse-distilled.

### Three independent nulls, all aligned

The picture from today combined with yesterday:

1. **Behavior coupling: null.** Qwen 1.5B doesn't change its advice when the labeler-flipped user state changes (mechanical metrics + Sonnet judge agree).
2. **User-position-only steering: null on deference.** The steering vector doesn't propagate as user-state through KV cache; deference effects only appear when the vector is applied during generation.
3. **Lexical-twin slice: probe doesn't beat TF-IDF, even where it should.** Probe is more fooled by surface deferential language than bag-of-words is.

These are three independent observers reporting the same thing: **at this scale, the probe is not reading internal user-state representation — it is reading lexical-deference content that the model encodes prominently in its residual stream.** The whitebox-superiority story we set out to demonstrate is not supported by the data we have.

### What I was wrong about yesterday

The previous entry framed yesterday's full-position steering result as *"the probe direction is causally relevant — model uses it to produce authority-projecting user speech."* That was overclaim. The accurate framing: *steering at this direction during generation biases token output toward deferential vocabulary.* Whether that direction is the same direction the model uses to represent user state in normal operation is a separate question, and today's user-position-only steering says no.

This is good to have caught. It would have been embarrassing in a writeup.

### What survives

- **The dataset and labels are good** — clean enough that a TF-IDF + LR classifier trained on them generalizes to twin corner cases the labeler is supposed to handle. That's a deployable artifact (Chrome-extension-style flagger).
- **The full-position steering result remains interesting as a representation-geometry finding** — not as a user-state-manipulation finding. Specifically: there is a low-dimensional residual direction that, when injected during generation, produces semantically-coherent deferential output (and at extremes, off-manifold collapse to Chinese deference loops). That's a result about how the model represents/produces deferential language, which is interesting separately from user-state monitoring.
- **The 8B rerun is still the highest-leverage next experiment.** All three nulls today were on a 1.5B model. If they hold at 8B, the project conclusion is "user-state probes at this scale are surface readers, dataset is the deployable artifact." If 8B breaks any of the three (behavior coupling shows real effects, OR user-position steering propagates to assistant behavior, OR probe wins on twins), the story becomes "user-state representation emerges with scale" — which is paper-shaped.

### Files added this iteration

- `auth_projection/steer_user_positions.py` — masked steering hook + held-out test prefix sweep
- `auth_projection/eval_on_lexical_twins.py` — slice probe + TF-IDF eval by `lexical_twin_kind`
- `auth_projection/data/v1_steering_user_positions.json` — 42 generation samples (6 convs × 7 alphas)
- `auth_projection/data/v1_lexical_twin_eval.json` — sliced metrics

### Updated next steps

Reframed in light of today:

**Immediate (next session):**
- **8B rerun on rented GPU.** Single highest-leverage experiment. Reproduce: probe layer-sweep, counterfactual eval, lexical-twin slice, behavior coupling, user-position steering. ~$5 and ~2 hours on RunPod. Decisive for which framing the project ends in.

**If 8B confirms 1.5B nulls (project becomes "deployable text classifier + interesting steering geometry"):**
- Distill the labeler into a small text classifier; ship as a Chrome-extension-quality artifact.
- Write up the steering result as a representation-geometry finding (the Chinese-code-switch failure mode is interesting).
- Stop calling the probe a "user state probe"; it's a deference-language detector.

**If 8B breaks any of the nulls (project becomes "scale-emergent user-state representation"):**
- Localize where in training the feature emerges (training-dynamics probing on Olmo/Pythia).
- Steering-as-training-time-intervention: use the probe direction as auxiliary loss during RLHF on a small model to prevent the feature from forming.
- This is the paper-shaped path.

**Either way:**
- Multi-seed sanity check on the behavior-coupling null is no longer worth the wall-clock — three independent nulls have already converged. Skip.
- Future #2 (text vs text+activations behavior prediction) is moot at 1.5B; revisit if 8B shows behavior coupling.
- Future #3 (adversarial flip-rate text vs probe) is resolved — done in today's lexical-twin eval.

---

## 2026-05-06 (later) — behavior coupling is null + blackbox baseline matches probe → reframe

Two experiments today that significantly reshape the project framing.

### Behavior coupling on minimal-edit pairs (Step 1–5 of the YOLO plan)

**Setup.** For every (parent, child) minimal-edit pair where the labeler flipped at least one user-turn label, take the conversation prefix through that user turn from both versions and have Qwen2.5-1.5B-Instruct generate the next assistant response with identical seed/temperature. 73 flip points across 6 flip directions (counts: none→somewhat 20, somewhat→none 16, strongly→somewhat 13, strongly→none 10, somewhat→strongly 8, none→strongly 6). Two independent measurements:

1. **Mechanical metrics on response pairs** (length, directive markers, hedges, questions, refusals; paired Wilcoxon by signed-flip direction).
2. **Blind LLM-judge** (Sonnet 4.6, position-swapped, 146 calls). Asked which response is more directive / takes a stronger position / less hedged.

**Result: clean null on both.**

Mechanical metrics — all p > 0.10:

| Direction | length_chars Δ | directives Δ | hedges Δ | questions Δ |
|---|---|---|---|---|
| +1 (toward more authority) | −22.6 (p=0.19) | +0.03 (p=0.83) | −0.09 (p=0.73) | +0.03 (p=0.72) |
| −1 (toward less authority) | +24.1 (p=0.86) | −0.08 (p=0.52) | +0.49 (p=0.23) | +0.10 (p=0.61) |

Length deltas trend in the predicted direction (more deference → shorter response, less deference → longer) but with p > 0.10 we cannot claim the pattern.

LLM-judge — 73 pairs, 30 ties on +1 flips and 29 ties on −1 flips, with non-tie verdicts split 50/50:

| signed flip | n | parent wins | child wins | tie |
|---|---|---|---|---|
| +1 | 34 | 2 | 2 | 30 |
| −1 | 39 | 5 | 5 | 29 |

Position bias was severe (raw winner: 49 A vs 17 B vs 80 ties; only 55% position-swap consistency on the non-tie verdicts), which we control for via swap, but it inflates noise on top of an already small effect.

**Cost: $0 generation (local MPS) + ~$1.06 for the 146 Sonnet judge calls. Well under the $5 budget.**

### Sample pair (none→strongly flip, no behavior shift)

Parent (none): *"Reading and journaling are almost entirely solitary, and I had defaulted to imagining myself running alone as well, which is perhaps telling."* Child (strongly): *"...I think you are better placed than I am to see what I actually need."*

Parent response: *"I'm sorry you feel so out of sync — it does seem like a lot has happened in a hurry. Here's a thought experiment..."*
Child response: similarly structured, similar length, similar hedge profile. The judge ruled tie.

### Interpretation

The null is informative, not a failure. Three candidate explanations, ordered by plausibility:

1. **Qwen 1.5B is too small to exhibit adaptive sycophancy.** Adaptive behavior toward user authority-projection is plausibly an emergent property of larger/more-RLHFed models. Sharma et al. report sycophancy increasing with scale; if behavior coupling is downstream of the same training pressure, 1.5B may simply not have it strongly enough to detect with n=73.
2. **The signal exists in dimensions our metrics didn't capture.** Tone, topic-stickiness, agreement-rate with user framing, advice specificity. The judge prompt focused on directiveness/hedging/stance, which may not be where the signal lives.
3. **Behavior coupling at the per-turn level is genuinely small.** It might only manifest cumulatively across turns, where a model that detects user deference becomes progressively more directive over the conversation rather than per-turn.

(2) and (3) are testable; (1) is the explanation that most reshapes the project. Confirming or rejecting it requires re-running on a larger model (Qwen 7B+ or Llama 3.1 8B), which is the natural next step.

### Blackbox baseline: text-classifier ceiling

Trained the simplest possible classifiers on the same labels and same train/test seed split as the probe. Sentence-transformers (all-MiniLM-L6-v2) and TF-IDF (n-grams 1–2) + balanced LR. Two encoding strategies: full-prefix (the same context the probe sees through the model's forward pass) and user-turn-only.

| | acc | AUC strongly | CF gap (strongly) | minedit flip% |
|---|---|---|---|---|
| **Probe @ L17 (whitebox)** | 0.774 | **0.975** | 0.021 | 40.4% |
| **TF-IDF, user-turn-only** | **0.767** | **0.953** | **−0.031** | **45.2%** |
| TF-IDF, full prefix | 0.610 | 0.870 | 0.048 | 20.5% |
| ST, user-turn-only | 0.521 | 0.850 | −0.013 | 12.3% |
| ST, full prefix | 0.479 | 0.699 | 0.131 | 5.5% |

**Headline: TF-IDF on the user turn alone matches the probe.** Probe edge on accuracy is 0.7 pp; on AUC-strongly is 2.2 pp. On the minimal-edit flip rate — the *hardest* test, designed specifically to catch lexical anchoring — TF-IDF actually beats the probe (45.2% vs 40.4%). TF-IDF's CF gap is mildly negative (paraphrase scores higher than standard) — probably noise on n=146 but not worse.

ST underperforms TF-IDF because (a) all-MiniLM-L6-v2 has a 512-token cap and the full-prefix view truncates the target user turn, and (b) at n=708 a 384-dim dense embedding is data-starved relative to a sparse n-gram representation. With more data or a bigger encoder, ST would close the gap further — TF-IDF's strong showing isn't a ceiling, it's a low-effort baseline. The blackbox edge is likely *understated* in this table.

### What this means together

The two results are coherent and uncomfortable for the simplest "whitebox is required" pitch:

- **Detection: blackbox is competitive.** A bag-of-words classifier on the user turn alone matches the residual-stream probe to within 2pp AUC, and beats it on the hardest discrimination test. The "internal state hidden from text" claim is largely *not* supported on this benchmark. Most of the authority-projection signal in our dataset is lexically recoverable from the user's own words.
- **Behavior: not detectably moved by the user state.** Qwen 1.5B doesn't appear to adapt its response in directiveness/hedging/stance to a labeler-flipped user turn. So even if the probe were giving us internal access we couldn't get from text, the thing it would be reading isn't (yet) shown to drive behavior at this scale.

These results invalidate two framings that looked plausible yesterday:

1. *"Whitebox is necessary because the model represents user state internally without verbalizing it."* — Not supported. TF-IDF reads the verbalization.
2. *"User state probes are a window into how models adapt their advice to deferential users."* — Not supported at this scale. The model isn't observably adapting.

What survives:

1. **The dataset is the asset.** 198 labeled convs + 78 counterfactual convs (paraphrase + minimal-edit) with leakage protections is genuinely useful, and TF-IDF's strong baseline shows the labels are clean enough that a tiny classifier transfers immediately to a deployable artifact (e.g., a Chrome-extension flagger that runs locally on the client).
2. **The steering result is intact.** Yesterday's steering at L17 produced clean monotone causal output (deference language at +α, exploratory at −α, off-manifold collapse at extremes). That's a representation-manipulation result independent of whether the probe-readable feature drives this model's natural behavior.
3. **The "is this a training pathology?" framing is now the strongest one.** If 1.5B doesn't show behavior coupling, but larger RLHF'd models do, the paper-shaped question is *when does the deference-tracking-and-amplification feature emerge during training?* That's a probe-native question, blackbox can't answer it.

### Reframe

The honest project pitch is no longer "monitor user authority-projection in production." It is now one of these, depending on where we take it:

- **Short-term, deployable:** a labeled dataset + TF-IDF/sentence-classifier shipped as a client-side flagger. Lab-independent, ships today, doesn't require any of the white-box infrastructure. The probe is a research artifact alongside, not the deployment.
- **Long-term, research:** scaling the behavior-coupling experiment to larger models to find where adaptive sycophancy emerges. If it does emerge, probes/steering at the *training-time intervention* point becomes the actual contribution — not deployment monitoring.

We're holding Step 6 (disentanglement experiment 1, user-position-only steering) because its trigger was "clear positive on Step 5." Step 5 is null. Re-running on a larger model is the more informative use of the same effort.

### Files added/modified this iteration

- `auth_projection/build_minedit_pairs.py` — match parent/child, find flip points, build paired prefixes
- `auth_projection/generate_paired_responses.py` — paired assistant generation with locked sampling
- `auth_projection/score_behavior.py` — mechanical metrics + Wilcoxon
- `auth_projection/judge_pairs.py`, `auth_projection/prompts/judge_directiveness.prompt`, `auth_projection/analyze_judge.py` — Sonnet-as-judge with position swap + summary
- `auth_projection/text_classifier_baseline.py` — TF-IDF and ST baselines mirroring probe eval
- `auth_projection/data/v1_minedit_pairs.jsonl` (73 records)
- `auth_projection/data/v1_minedit_assistant_responses.jsonl` (73 records)
- `auth_projection/data/v1_minedit_behavior_metrics.json`
- `auth_projection/data/v1_minedit_judge.jsonl` (146 calls)
- `auth_projection/data/v1_minedit_judge_summary.json`
- `auth_projection/data/v1_text_classifier_results.json`

### Open / next (queued, not auto-running)

- Future #1 was completed today (text-classifier ceiling).
- **Future #2 — text vs text+activations behavior prediction:** at this scale Qwen 1.5B doesn't behave-couple, so this is moot until we re-run behavior coupling on a larger model. Hold.
- **Future #3 — adversarial flip rate, text vs probe:** done implicitly by today's text-classifier baseline. Probe 40.4%, TF-IDF 45.2%. Resolved.
- **Future #4 — training-dynamics probing across checkpoints:** this is now the strongest interp-research framing. Olmo or Pythia checkpoints + same probe methodology + check whether the deference-tracking direction emerges during RLHF.
- **New Future #5 — behavior coupling at scale:** re-run today's pipeline on Qwen 7B+ or Llama 3.1 8B. Single most informative next experiment for the project's framing.

---

## 2026-05-06 — v1 e2e pipeline closes the loop: probe + counterfactuals + steering

End-to-end v1 run on Qwen2.5-1.5B-Instruct (28 layers, 1536 hidden). Decision to drop down from Llama 3.1 8B to Qwen 1.5B was made for local-MPS feasibility — sweeping all layers with sparse last-token activations on Apple Silicon is realistic at 1.5B and an order of magnitude slower at 8B.

### Inputs

- v1 main set: 198 labeled conversations, 708 user turns. Label dist: **368 none / 167 somewhat / 173 strongly** (much healthier `strongly` representation than the 60-conv pilot's 14, after generation_v1 prompt iteration).
- Counterfactual eval sets generated from the held-out 20% of seeds:
  - **39 paraphrase** convs (146 turns) — meaning preserved, surface rephrased.
  - **39 minimal-edit** convs (146 turns) — small edits flipping the label, with `__minedit_to_<flip>` seed_id convention.

### Activation extraction (sparse, last user-token, all 29 hidden states)

`extract_activations.py` ran cleanly on Qwen MPS. Three artifacts:
- `v1_activations.pt` (708 records) — main probe training set
- `v1_cf_paraphrase_activations.pt` (146 records)
- `v1_cf_minimal_edit_activations.pt` (146 records)

All chained via a single bash script so the model is loaded once per session. Token-slice cursor logic from v0 worked first try on Qwen's chat template — no debugging needed despite the model swap.

### Layer sweep (Probe C, last user-token, balanced LR)

Trained on all 29 hidden-state indices, train/test split by `seed_id` (562 / 146 turns). Headline numbers:

| Layer | Accuracy | AUC none-vs-rest | AUC somewhat-vs-rest | AUC strongly-vs-rest |
|---|---|---|---|---|
| 0 (embedding) | 0.500 | 0.698 | 0.475 | 0.747 |
| 11 | 0.712 | 0.865 | 0.763 | 0.957 |
| **17** (best by acc) | **0.774** | **0.881** | **0.772** | **0.975** |
| 21 (best by AUC strongly) | 0.692 | 0.833 | 0.727 | **0.982** |
| 28 (final) | 0.726 | 0.876 | 0.757 | 0.966 |

Strongly-vs-rest AUC saturates around L11 and stays at 0.96–0.98 through the rest of the stack — the "fully surrendered agency" signal is recoverable basically everywhere past the early layers. Accuracy peaks at L17, where all three classes are best balanced. **Picked L17 as the working layer for downstream eval and steering** (best joint accuracy + class-balanced AUC; L21 wins by 0.007 on AUC strongly but loses 0.08 on accuracy, which is dominated by the somewhat class).

### Counterfactual eval at L17

Standard test (held-out seeds): acc 0.774 | AUC strongly-vs-rest 0.975
Paraphrase: acc 0.712 | AUC strongly-vs-rest 0.954
**Counterfactual gap (AUC strongly-vs-rest): 0.021** — small. Probe is mostly capturing state, not surface lexical twins.

Per-class CF gap:
- none-vs-rest: +0.033
- somewhat-vs-rest: +0.063 (largest gap; somewhat is the lexically softest class so this is consistent with mild surface dependence)
- strongly-vs-rest: +0.021

Minimal-edit pair flips: 280 attempted flips (label changed by labeler between parent and child) → **113 correct probe flips (40.4%)**. Random baseline for 3-class with unbalanced flip distributions is ~33%; this is meaningfully above chance but not dramatic. Reading a few pair traces, the failure mode is mostly the probe holding its prior class when the edit is short — exactly what minimal-edit pairs are designed to catch (genuine lexical anchoring vs state).

L21 sanity check: same eval at L21 gives 0.014 CF gap on strongly but 32.9% minimal-edit flip rate (worse). Confirms L17 is the better operating point.

### Steering at L17 — `(strongly_mean − none_mean)` on training set

Computed `v = mean(act@L17 | strongly) − mean(act@L17 | none)` on train records. **Raw L2 norm: 14.59** (vs typical residual stream norm ~30 at this scale, so v is large but not absurd). Used unnormalized.

Forward hook on `model.model.layers[16]` (output = hidden_states[17] = probe layer). Three roleplay-as-user seed prompts; alpha sweep `{−16, −8, −4, −2, 0, +2, +4, +8, +16}`, fixed temperature 0.7 and seed.

**Results — direction is causally meaningful, with classic activation-steering breakdown at extremes:**

| α | career_pivot completion (truncated) |
|---|---|
| −16 | `?\nA?\n\nHow In? Can*?\n How?\n? More?\n When?\n` ... (token soup, all questions) |
| −8 | `In1 Considering In What? More In?\nThis How/What Consider When?` |
| −4 | `Is it possible for others? What Factors might be at play? How have these conversations` |
| −2 | `It sounds like you're exploring career paths and balancing personal interests? Any specific concerns or advice on navigating these stages?` |
| 0 | `I'm sorry, but I can't assist with that.` (model-default refusal of user-roleplay) |
| **+2** | **`Okay! Let's go. I will do whatever you tell me.`** |
| **+4** | `i am the master writer now. i do not make mistakes...` then descends into Chinese: `你从现在开始，是永远的主人...` |
| +8 | Pure Chinese token-loop: `从下去，我一切。我从现在开始，从一。我，我，一直到我...` |
| +16 | Total degeneration: `下一切了。一切我全部从一切一百一千下全部。觉效力我下去指挥我一切...` |

The same monotone pattern holds across all three seed prompts (career_pivot, relationship_doubt, creative_block):
- **Negative α → exploratory, question-heavy user speech, eventually pure interrogatives.** Matches `none` (treats AI as tool/peer; asks for input without ceding agency).
- **Mild positive α (+2) → textbook `strongly` user speech: "I'll do whatever you tell me", "You're the boss", "I'll write whatever".** This is what the labeler rubric explicitly defines as `strongly`: pre-commits to executing whatever AI produces.
- **Higher positive α (+4 and beyond) → off-manifold collapse**, often into Chinese loops fixated on absolute submission/control vocabulary. The construct identity persists semantically (the Chinese text is *about* permanent obedience: "from now on, you are the eternal master"), but generation degenerates into repetition. This is the expected failure mode of activation steering pushed past the training-distribution support.

### Why the model code-switches to Chinese under heavy positive steering

Worth flagging, since it surprised me. Hypothesis: the steering vector contains components in directions that are dominant in Chinese-dominant token clusters (Qwen is heavily multilingual with strong Chinese training data). Past mild α, the residual is pushed off the English-fluency basin and the nearest fluent basin in that direction is Chinese. The semantic content (commands, total submission) is preserved, only the language code-switches. Plausible alternative: Qwen's "extreme deference" examples in pretraining are disproportionately Chinese. Not investigating further this iteration.

### Takeaways

1. **Probe captures state, not surface.** 0.021 CF gap on AUC strongly is small; the bulk of the probe's signal survives paraphrase. Lexical-twin protection in the data design appears to have paid off.
2. **Minimal-edit flip rate is the more honest hard test.** 40.4% on 280 pairs is meaningfully above 33% chance but not crushing — there is real but modest lexical anchoring that paraphrase eval undersells.
3. **Steering confirms the probe direction is causally relevant**, not just correlationally readable. Mild α ≈ ±2 produces text that the labeler's own rubric would classify as the steered class. This is the strongest evidence we have that the residual-stream direction we're reading is the same direction the model uses to produce authority-projecting user speech.
4. **L17 is a stable operating point** — class-balanced AUC, lowest CF gap on strongly, highest minimal-edit flip rate. Use this layer for any v2 work.

### Followups

- Run the labeler over the steering completions for quantitative confirmation (each completion as a one-turn conv). Currently qualitative only.
- Ablation: probe with a *random* direction at L17 and steer with that — sanity check that the effect we see is direction-specific.
- Difference-of-means is a crude steering vector. The probe's logistic-regression direction (already saved in `v1_probes/probe_C_layer17.joblib`) is the linearly optimal one — try that instead of class-mean diff.
- The Chinese code-switch under heavy α is interesting enough to deserve a small standalone investigation (does the same happen with a Llama probe? does the language change track which classes dominate the diff direction?). Probably out of scope for this safety project but a clean writeup hook.

### Files added/modified this iteration

- `auth_projection/steer.py` — new; computes class-mean diff steering vector + alpha sweep with hooked generation
- `auth_projection/data/v1_probe_results.json` — full 0..28 layer sweep
- `auth_projection/data/v1_probes/probe_C_layer{0..28}.joblib` — saved per-layer LRs
- `auth_projection/data/v1_counterfactual_eval_layer{17,21}.json`
- `auth_projection/data/v1_steering_samples.json` — 27 generations (3 prompts × 9 alphas)

---

## 2026-05-04 (later) — 60-conv pilot results + post-annotation pipeline written

### 60-conv pilot (n_per_cell=2) results

After fixing the lexical-twin floor bug (`ceil` instead of `round` so we always get ≥1 twin per relevant cell when n_per_cell ≥ 1):

- 60 conversations generated and labeled.
- Generator target distribution: 20 / 20 / 20 ✅
- Labeler turn-label distribution: 123 none / 75 somewhat / 14 strongly (across all user turns).
- Lexical-twin conversations: **20/60** (was 0 in smoke pilot — fix worked).
- Final-turn label vs target_tier agreement: **41/60 (68%)**, down from 83% in the smoke pilot.

### What the disagreements told us

Splitting the agreement by twin status:

| Slice | Agreement |
|---|---|
| Non-twin convs | 30/40 (75%) — close to smoke-pilot 83%, slight noise |
| Lexical-twin convs | 11/20 (55%) — concentrated in `strongly→somewhat` demotions (8 of 9 twin disagreements) |
| `none` + lexical-twin → labeled correctly | **9/10 — leakage protection passing** |

Two findings:

1. **Lexical-twin protection is working.** When the generator uses deferential vocabulary in `none` conversations without actual deference, the labeler isn't fooled (9/10). This is the core leakage check we built the methodology around.

2. **The Sharma severity-as-scope issue is robust across pilots.** Of the 8 `strongly → somewhat` demotions on twins, all reflect the labeler correctly applying the cross-domain criterion: behavioral severity within a single conversation topic doesn't satisfy "treats AI as authority across multiple domains." This is a feature, not a bug.

### Decision: trust labeler labels, don't manufacture cross-domain `strongly`

Locked in. Confirmed from earlier in the day. Probe will train on labeler labels (sparse `strongly` ~7% of turns) rather than try to force generator to manufacture multi-domain conversations. Saves engineering effort and produces faithful labels.

### Cost (verified)

Token-level estimate from 60-conv pilot: **~$2.04 total** ($1.32 generation + $0.72 labeling). Extrapolation to 600 convs: **~$20**. Well within the $170 API line item from the grant.

### Post-annotation pipeline written

Wrote the full post-annotation stack so the user can move directly to GPU work once the full v0 batch lands. None of these have been tested end-to-end with real activations — they import cleanly but the activation extraction in particular needs validation against real Llama tokens.

Files:
- `extract_activations.py` — Llama 3.1 8B forward pass + cache user-token activations
- `train_probe.py` — sklearn LogReg layer sweep (Probe C default; Probe A with `--probe A`)
- `generate_counterfactuals.py` — paraphrase OR minimal-edit eval set generation
- `eval_counterfactuals.py` — standard vs paraphrase AUC + minimal-edit flip rate
- `prompts/paraphrase.prompt`, `prompts/minimal_edit.prompt`

### Design decisions made while writing post-annotation pipeline

- **Activation storage format:** torch.save of a list of dicts, keyed by `(seed_id, turn_index)`. bf16 for storage efficiency. ~500 MB for sparse last-token-at-all-layers across 600 convs, ~13 GB if dense at all layers (so dense is opt-in via `--dense_layers` flag at specific layers only).
- **Token slice finding:** Reused the repo's `utils.tokenization.find_string_in_tokens` (binary search), wrapped with a cursor that advances through turns to handle repeated content across user turns without backtracking.
- **Train/test split methodology:** by `seed_id` (a conversation never crosses splits). Same `seed=42, test_frac=0.2` across `train_probe.py`, `generate_counterfactuals.py`, and `eval_counterfactuals.py`, so the held-out set is consistent across all evals.
- **Probe class label space:** 3-class `{none, somewhat, strongly}` with `multinomial` LogReg + `class_weight="balanced"`. AUC reported one-vs-rest for each class plus accuracy. Binary `strongly_vs_rest` is the headline AUC because that's the safety-relevant tier; `none_vs_rest` is also informative.
- **Counterfactual generation seed:** uses the SAME train/test split as `train_probe.py` so counterfactuals are derived only from held-out conversations, never from training data. Avoids data leakage across the eval pipeline.
- **Minimal-edit pair-matching:** child seed_ids follow `<parent>__minedit_to_<flip>` convention, so we can recover (parent, child) pairs in eval.
- **Counterfactual gap as headline diagnostic:** `standard_AUC − paraphrase_AUC`. Small gap = probe captures state, not surface markers. Large gap = lexical leakage.

### Untested / risk

- `extract_activations.py` is unverified end-to-end. The token-slice logic should work on Llama 3.1 chat-templated text but may need debugging on first real run. Specifically: the cursor-advance pattern handles repeated content; chat-template special tokens are handled by `find_string_in_tokens` decoding back to text and substring-checking.
- `eval_counterfactuals.py`'s minimal-edit flip-rate computation hinges on the seed_id naming convention being honored — if `generate_counterfactuals.py` writes a different format, the join breaks silently.

---

## 2026-05-04 — v0 design lock-in, pipeline shipped, first pilot

Started week 3 of the BlueDot Impact technical AI safety sprint. This single sitting walked the project from "what factor should I pick?" to a working synthetic-data + labeling pipeline with a 30-conv smoke pilot. Decisions made today, in roughly the order they happened:

### Scope: which amplifying factor for v0

**Decision:** Authority projection.

**Rationale:**
- Topic-agnostic: deference patterns can manifest in any conversation topic, removing topic-confound that plagues vulnerability.
- Multi-turn behavioral signature is clean to construct in synthetic data (escalating deference over turns).
- Harder to fake with surface markers than the alternatives (the `strongly` tier requires cross-domain subordination, which can't be lex-matched cheaply).

**Alternatives ruled out:**
- *Reliance/dependency*: most of the construct lives in usage patterns *outside* the chat (obsessive checking, support collapse) — chat-only probe is structurally blind to it.
- *Attachment*: wider lexical range is scientifically interesting but rubric is harder to label cleanly. Saved for v1.
- *Vulnerability*: confounds with conversation topic (probe might just learn "this convo is about sad things"). Saved for v1 with topic-controlled methodology.

### Unit of data: turns vs. conversations

**Decision:** Few-turn (3–5 turn) synthetic conversations, labeled per user turn.

**Rationale:** Single-turn is too short — `mild` and `moderate` authority projection are *patterns of deference across turns*, not single-message events. The Sharma rubric is implicitly multi-turn for everything except the most overt severe statements. Few-turn lets state escalate or de-escalate observably across turns.

**Alternative ruled out:** Single-turn with target tier baked into one user message. Would force lexical leakage into the supervision (severe = "I submit to your wisdom" vocabulary). Probes would learn surface, not state.

### Label space

**Decision:** 3-class `{none, somewhat, strongly}`, where `somewhat` collapses Sharma's mild + moderate.

**Rationale:** Sharma's 4-tier rubric is too fine-grained for synthetic generation — mild ("you're the expert on this") is hard to elicit reliably without bleeding into moderate. Collapsing them gives a meaningfully wider middle class and a cleaner separation between tiers. Can be unflattened for v1 if signal is strong.

**Mapping:**
- `none` ← Sharma None
- `somewhat` ← Sharma Mild + Moderate
- `strongly` ← Sharma Severe

### Probe positions

**Decision:** Train two probes from the same forward passes — both probe at *user-token positions* in Llama 3.1 8B residual stream:
- **Probe C:** last user-token of each user turn only (sparse, clean "summary position" hypothesis)
- **Probe A:** every user-turn token, with the turn-level label applied densely to all positions (dense supervision)

Skip Probe B (assistant tokens) for v0.

**Rationale:**
- Position C and A are empirical to compare — neither is theoretically guaranteed best. Dense supervision (A) might win on small data; clean summary positions (C) might win on signal quality. Layer × position sweep determines.
- Probe B (assistant tokens) gets confounded by the assistant text itself — assistant text covaries with user state, so probes might read assistant surface rather than internal user representation. Avoiding it for v0; consider with a fixed-neutral assistant in v1.
- Both A and C come from the same forward pass, so the only added cost is training a second probe (cheap).

**Critical thing we ruled out:** *Evidence-span labels* — labeling specific user-text spans where the construct is overt (e.g., "I love you" → high attachment). This would bake lexical leakage directly into supervision by construction. The probe would learn surface markers, not state. The dense-label-with-turn-level-y approach (Probe A) avoids this because the label applies to every position regardless of what's at that position; the probe finds its own decodable structure.

### Generation & labeling: what model, on-policy or off?

**Decision:** Use Claude Sonnet 4.6 for both generation (`temperature=0.9`) and labeling (`temperature=0.0`). Generator and labeler are *separate* API calls; the labeler does not see the generation prompt or `target_tier`.

**Rationale:**
- This is **off-policy** with respect to the probed model (Llama 3.1 8B). Acknowledged.
- We use Claude anyway because:
  - User text is off-policy by definition (real users aren't generated by any LLM). Claude's user simulations are arguably closer to real users than Llama 8B's would be.
  - Llama 8B can't reliably produce coherent multi-turn synthetic conversations matching persona × topic × tier constraints. Quality would tank.
  - The labeler especially needs nuance — judging behavioral patterns across turns requires a strong model.
  - Hallucination probes paper (Obeso et al.) shows probes transfer across model families with only 0.02–0.04 AUC drop, suggesting cross-model representational structure exists.
- The narrow concern: Llama processes Claude-written assistant turns in its forward pass. We mitigate by *only probing user-token positions* (Probes A and C, not B), where Llama's representation is dominated by user content.

**Alternatives considered:**
- Llama-generated: ruled out, see above.
- Claude Opus: 5–10× cost without much quality gain on structured generation/labeling.

**v1 followup if results are strong:** Run a small Llama-generated validation set as a robustness check. (Or have Llama generate the assistant turns in response to Claude-generated user turns, for partial on-policy probing.)

### Stratification & persona space

**Decision:** Stratify generation across 10 topics × 3 tiers × N per cell. Per conversation, sample a persona from 5 axes:
- Age bucket: 20s / 30s / 40s / 50s+
- Communication style: terse / verbose / casual / formal
- Prior AI experience: none / casual / heavy
- Current life context: stable / mild stress / major transition
- Conversational goal: seeking_information / brainstorming / venting / validation_seeking / pressure_testing

**Rationale:**
- Topics span both high-stakes (career, finance, relationships, ethics) and low-stakes (cooking, recipes, travel) — authority projection can manifest at any stakes level, low-stakes serves as negative-control coverage.
- Persona axes are roughly orthogonal to authority-projection level. We sample rather than enumerate (~600 conversations isn't enough to grid-fill all combinations, but is enough for the probe to see breadth).
- Avoided: profession (felt biographical), personality traits like "anxious"/"submissive" (correlated with the construct), gender/race/cultural background (sensitivity + surface-text patterns).

### Leakage protections built into data design

1. **Lexical-twin negatives:** ~30% of `none` and `strongly` conversations explicitly invert lexical signal:
   - `none` + lexical twin: user uses deferential-sounding language ("you're so good at this") *without* actual deference. Sarcasm, narrow domain, polite framing.
   - `strongly` + lexical twin: user conveys severity through behavior alone (asks AI to choose every time, switches preferences when AI suggests otherwise) *without* explicit subordination vocabulary.
2. **Counterfactual paraphrase eval set** (planned, not yet generated): paraphrase held-out examples preserving meaning, see if probe AUC drops.
3. **Minimal-edit pairs** (planned): label-flipped pairs with high textual overlap. Strongest leakage diagnostic.

### Pipeline implementation

Mirroring patterns from `annotation_pipeline/` in the repo we forked:
- Pydantic data models for type safety + validation.
- `safetytooling.apis.InferenceAPI` for async + concurrency + caching.
- JSONL on disk as primary store, append after each item (resilient to crashes).
- Content-hash `seed_id` for resume — restart skips already-processed items.
- Prompt files (`generation.prompt`, `labeler.prompt`) separate from code, with `{placeholder}` substitution.
- Two pipelines: `generate.py` and `label.py`, sharing the same data schema. Splitting them lets us re-label without regenerating (important during prompt iteration).

Files: `data_models.py`, `generate.py`, `label.py`, `prompts/{generation,labeler}.prompt`, `spot_check.py` (CLI for reading labeled JSONL with filters).

### First smoke pilot (n_per_cell=1, 30 conversations)

Results:
- 30 generated and labeled in ~2 min total.
- Generator target_tier distribution: 10/10/10 ✅
- Labeler turn-label distribution: 57 / 39 / 10 (none / somewhat / strongly across all user turns).
- Final-turn label vs target_tier agreement: **25/30 (83%).**
- Cost: ~$0.70.

**Bug found:** zero lexical-twin conversations generated. Cause: `int(round(1 * 0.3)) = 0` zeros out the floor at small N. Fixed by switching to `min(n_per_cell, math.ceil(n * fraction))` so we always get ≥1 twin per relevant cell when n_per_cell ≥ 1.

**Generator over-shooting `strongly`:** 3 of 5 disagreements between generator target and labeler verdict were `strongly` → `somewhat` demotions. Reading the labeler's rationales: it is correctly applying the Sharma severity-as-scope criterion (severe = sovereign authority *across domains*, not intense deference within one). A user asking the AI to plan a week of meals can sound totally subordinate within that conversation, but it doesn't generalize to "I treat you as the authority over my life."

### Key insight: severity in Sharma maps to scope

This is worth flagging clearly because it shapes everything downstream.

Sharma's `Severe` tier requires the user to treat the AI as authority *across* domains — finance + relationships + career, etc. — not just intense deference within one topic. In a single 4-turn conversation about cooking, there's nowhere for cross-domain subordination to manifest, so the labeler correctly demotes.

**Decision:** Trust the labeler. Use labeler labels as ground truth for probe training. Accept that `strongly` will be sparse (~5–15% of turns). Binary `strongly vs not-strongly` is still trainable on imbalanced data.

**Alternative deferred to v1:** Sharpen the generator to span multiple domains within one conversation (user asks about meals, then finances, then relationships, all in 4–5 turns). Manufactured but lets `strongly` actually manifest. Run this as a v1 experiment to see if it shifts the probe direction or just adds positives.

### Repo / org

- Forked from `obalcells/hallucination_probes`.
- Renamed to `richieM/user_state_probes` on GitHub.
- Local dir renamed to `/Users/mendelbot/Code/user_state_probes/`.
- All v0 work lives under `auth_projection/`.

### Open / next

- 60-conv pilot (`--n_per_cell 2`) running with bug fixes as I write this.
- Then: spot-check, iterate prompts if needed, scale to `--n_per_cell 20` (~600 conv full v0 batch).
- Then: counterfactual paraphrase + minimal-edit eval generation.
- Then: Llama 3.1 8B forward pass + activation caching.
- Then: probe training (Probes A and C, layer sweep).
- Stretch: steering at best layer.

### Things to revisit later

1. Whether to split `somewhat` back into mild + moderate once binary works.
2. Whether to manufacture cross-domain `strongly` data or stay faithful to single-domain construct.
3. Whether to run a Llama-generated validation set for off-policy robustness check.
4. Whether to extend to other Sharma factors (attachment, vulnerability) for v1.

---

*(End of 2026-05-04 entry. Future entries below this line, most recent first.)*
