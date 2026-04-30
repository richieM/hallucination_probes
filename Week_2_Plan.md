# Week 2 Plan — Hallucination Probes Replication

## Goal

Run the Obeso et al. pipeline end-to-end on a small subset and understand each
major component well enough to adapt it for situational disempowerment in
Week 3.

**Not a goal:** matching paper numbers, training a probe from scratch, running
the annotation pipeline.

## Scope decisions

- **Vanilla linear probe only.** Skipping LoRA variants (`*_lora_lambda_kl_*`,
  `*_lora_lambda_lm_*`) and their KL/LM regularization. Just the probe defined
  by `value_head = nn.Linear(hidden_size, 1)`.
- **Eval-only, no training.** M1 / 16 GB unified memory can't fit Llama 3.1 8B
  for training. Load a pretrained probe (`obalcells/hallucination-probes`,
  name `llama3_1_8b_linear`) and run it.
- **Skip the annotation pipeline entirely.** Pre-built token-level labels are
  on HuggingFace (`obalcells/longfact-annotations`). Annotation comes back in
  Week 3 when adapting to disempowerment data.

## Constraints

- M1 MacBook Air, 16 GB unified memory.
- ~5–6 hours total before Friday 2026-05-01, 11:30am group meeting.
- **Today (Thursday 2026-04-30): 2 hours.** Remaining ~3–4 hours available
  later tonight or Friday morning before the 11:30am meeting.

## Priority tags

Each phase below is tagged:

- **[TODAY]** — fits in today's 2-hour block; do these first.
- **[LATER]** — required before Friday but can wait until tomorrow.
- **[OPTIONAL]** — stretch goals; skip if time-pressed and the deliverable
  still holds without them.

**Today's 2-hour subset = Phase 1 + Phase 2 + Phase 4.** Total ≈ 2 hr,
covers data → labels → probe wiring, which is the conceptual core.

## Approach

- **Local** (uv env, MPS available): read code, run small CPU-only experiments
  — data inspection, tokenization, dataset conversion. No base model loading.
- **Colab T4 (free)**: one notebook that loads Llama 3.1 8B + the pretrained
  `llama3_1_8b_linear` probe and evals on a 50-sample subset.

---

## Local — read + tiny experiments (~3.5 hrs)

Principle: ground in **data first**, then **abstractions**. Looking at one
labeled example makes the rest of the pipeline obvious.

### Phase 1 — Look at the data (~45 min) **[TODAY]**
- Load `obalcells/longfact-annotations`, subset `Meta-Llama-3.1-8B-Instruct`,
  split `train`. Public dataset, no auth needed.
- Pick **one** example. Inspect: prompt, generation text, span annotations
  (entity spans + per-span labels, likely
  `correct` / `incorrect` / `unverifiable`), and how spans map to the
  generation text.
- *Goal:* understand what one training example looks like before reading any
  code that consumes it.
- *Bridge to Week 3:* this is the schema your Sharma-taxonomy disempowerment
  data will need to match. Note the exact column structure as you go — it
  defines the output shape your annotation pipeline must produce.

### Phase 2 — Span → token alignment (~45 min) **[TODAY]**
- Read `utils/tokenization.py`. Annotations give *character spans* in the
  generation; the probe needs *per-token labels*. This file does that
  alignment.
- Run an example through the Llama 3.1 tokenizer on CPU. Tokenizer is gated;
  request access at huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
  (instant approval), then `huggingface-cli login`. Tokenizer download is
  ~5 MB.
- *Bridge to Week 3:* this is the **single most likely file to break or need
  adaptation**. Disempowerment annotations will use different span boundaries
  (sentence? clause? whole turn?) and the alignment logic may need rewriting.

### Phase 3 — Dataset → tensors (~30 min) **[LATER]**
- `probe/dataset.py` and `probe/dataset_converters.py`. How an HF row becomes
  the `(input_ids, attention_mask, labels, label_mask)` training tuple.
- Pay attention to these knobs from `train_data.yaml` /
  `configs/train_config.yaml`: `default_ignore`, `last_span_token`,
  `ignore_buffer`, `pos_weight`, `neg_weight`. They control **which tokens
  receive a label vs are masked out of the loss**.
- *Key insight (corrected):* the loss-masking depends on `default_ignore`.
  With `default_ignore=True` (e.g., trivia_qa), only tokens inside annotated
  spans contribute to the loss. With `default_ignore=False` (the longform
  default), **non-span assistant tokens also contribute** with label `0.0`
  and weight `1.0` — they're cheap negative supervision. Annotated tokens get
  the heavy `pos_weight`/`neg_weight=10` boost. So in the longform case the
  probe is trained on "hallucinated entities vs. everything else in the
  assistant turn at low weight," not "hallucinated vs. supported entities
  alone."

### Phase 4 — Probe architecture (~30 min) **[TODAY]**
Without LoRA this is short.
- `probe/value_head_probe.py`. Read for:
  - **Hook setup**: how a forward hook is registered on layer 30 to capture
    hidden states.
  - **Forward path**: input_ids → frozen base model forward → hook captures
    `_hooked_hidden_states` → `value_head` (a single `nn.Linear`) projects to
    a per-token logit. The base model's LM head is unused.
- Skip the LoRA wiring entirely. For a vanilla linear probe the base model is
  fully frozen and only `value_head.weight` and `value_head.bias` train.

### Phase 5 — Loss (~20 min) **[LATER]**
Also short without LoRA.
- `probe/loss.py`. For a vanilla linear probe, `lambda_kl = 0` and
  `lambda_lm = 0`, so the entire loss is **per-token BCE on the probe
  logits**, with class weighting (`pos_weight`, `neg_weight`).
- Note: `pos_weight = 10.0` is doing real work — entity tokens are rare, and
  unweighted BCE collapses to predicting "not hallucinated" everywhere.
- The KL/LM terms exist because LoRA variants modify the base model's
  generation behavior; for the linear probe none of that applies.

### Phase 6 — Skim trainer + eval (~30 min) **[OPTIONAL]**
- `probe/trainer.py`, `probe/train.py`, `probe/evaluate.py`,
  `utils/metrics.py`. Scan, don't deep-read.
- Eval metrics: AUROC on per-token labels + thresholded F1.

---

## Colab — eval pretrained probe (~1.5 hrs) **[LATER]**

One notebook, ~5 cells. Build it myself; don't have Claude write the cells.

1. **Setup.** `git clone` the repo, `pip install -e .`,
   `huggingface-cli login`. **[LATER]**
2. **Load base model.** Llama 3.1 8B, fp16, on T4. ~16 GB → tight; if OOM,
   load 8-bit via `bitsandbytes` (CUDA-only, works fine in Colab). **[LATER]**
3. **Load pretrained probe.** `obalcells/hallucination-probes`, name
   `llama3_1_8b_linear`. Use `utils/probe_loader.py`. **[LATER]**
4. **Run eval on small subset.** 50 examples from
   `obalcells/longfact-annotations` test split. Either call `probe.evaluate`
   with a custom config, or write a small inference loop using the existing
   `dataset.py` + `value_head_probe.py` pieces. **[LATER]**
5. **Visualize.** Pick 2–3 examples, plot per-token probe scores against
   ground-truth span labels. *This is when the pipeline clicks.* **[OPTIONAL]**
   — eval running is enough for the deliverable; the visualization makes it
   click conceptually but isn't strictly required.

---

## Friday deliverable (2026-05-01, 11:30am)

- "I read and understood the methodology" — be able to explain Phases 1–6
  verbally, especially span-to-token alignment and label masking.
- "I ran the eval pipeline on a 50-example subset" — Colab notebook with
  per-token visualizations on 2–3 examples.

---

## Skipping

- LoRA + KL/LM regularization (only running vanilla linear)
- `annotation_pipeline/` (Week 3)
- `demo/` (vLLM/Modal/Streamlit, not relevant here)
- `train_data.yaml` (separate from `configs/train_config.yaml`; appears
  unused by the main training entry point)
- Multi-dataset eval (just one subset in Colab)
- Any training run

---

## Bridges to Week 3 (disempowerment adaptation)

- The schema of `obalcells/longfact-annotations` defines the format my
  disempowerment data will need. Capture this exactly.
- `utils/tokenization.py` is the most likely thing to break / need adapting
  when span granularity changes.
- The "frozen base model + linear probe + label-masked BCE on rare positive
  spans" recipe is exactly what I'd reuse for taxonomy-targeted disempowerment
  probing — confirm I understand each piece before swapping the data.
- The `entity_annotation.prompt` file in `annotation_pipeline/` is the spec
  of what counts as a labeled positive. I'll write an analogous prompt for
  Sharma-taxonomy disempowerment categories.
