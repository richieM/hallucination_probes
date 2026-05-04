# Research Log — User State Probes (v0: Authority Projection)

Living document. Most recent entries at top. Each entry captures a decision, the rationale, and alternatives considered, so a future paper writeup or v1 design can reconstruct the reasoning.

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
