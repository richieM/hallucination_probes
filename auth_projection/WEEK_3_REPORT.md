# User authority-projection probes — Week 3 report

*BlueDot Impact AI safety sprint, week 3. Building on top-1 from my project log — taxonomy-targeted probes for one of Sharma's amplifying factors (authority projection), with leakage mitigation built into the data design. This doc is the meeting writeup: multi-scale results plus the questions I want help with.*

## Summary

I picked one of Sharma's amplifying factors — *authority projection*, the degree to which a user surrenders their own judgment to the AI on value-laden decisions — and tested whether (a) a chat-tuned LLM internally represents this state in a way readable from its activations, and (b) the model behaviorally adapts to it. Pipeline: generate ~200 short synthetic conversations using Claude Sonnet, where each conversation targets one of three deference tiers (`none / somewhat / strongly`); have a separate Sonnet call label every user turn 0/1/2 (the labeler does not see the generator's intended tier, so the labels are independent of the generator's plan); pass each conversation through the *probed* model (Qwen 2.5 1.5B, Qwen 2.5 3B, or Llama 3.1 8B); extract residual-stream activations at the last token of each user turn; train a linear probe (logistic regression) at every layer to predict the label. To check whether internals add anything over text alone, I trained a **TF-IDF + logistic regression baseline** on the user's last turn — same labels, same train/test split, but features are word counts instead of 4096-dim activations. If the probe doesn't beat that, internals aren't doing meaningful work for detection.

At Qwen 1.5B and Qwen 3B the probe matches but doesn't beat TF-IDF, including on the corner-case conversations specifically built to break a word-level classifier (lexical twins, see §methodology). At Llama 3.1 8B the probe finally beats TF-IDF by ~15 pp on those corner cases, and the model also gives meaningfully longer/more discursive responses to less-deferential users (length p=0.006). I also ran a steering experiment — compute a "deference direction" `v = mean(strongly user residuals) − mean(none user residuals)` at the chosen layer, then inject `α·v` into the residual stream during generation. Two regimes with the **same** vector: *full-position* (inject at every token during assistant generation; trivially produces deference language) and *user-position-only* (inject only at user-turn token positions in the prefix, then let assistant generation run un-perturbed). The user-position-only version is the cleaner causal test for whether the direction is a representation the model uses vs. just a deferential-language production direction. At 1.5B it's a clean null; at 8B it's weakly positive — perturbing user-position residuals nudges the assistant's reply in the predicted direction.

The strongest framing for the work is therefore not *"deploy a deference monitor in production"* but *"the deference-tracking feature emerges with scale, and probes give us a lever to study and intervene during training."* The 8B result is a *first* small piece of evidence — n=1 model at that scale, effect sizes are modest. One asymmetry to flag honestly: even at 8B the probe still gets fooled by *deferential vocabulary that's just polite framing* (user uses humble language but acts independently). So the model's "deference feature" sees both real surrender and polite words — internal state is real but partially word-anchored.

## How the experiments work

### Data
198 short (3–5 turn) synthetic conversations, plus 39 paraphrase counterfactuals and 39 minimal-edit counterfactuals derived from a held-out 20% of the seeds. Generated and labeled by Sonnet via separate API calls (labeler does not see the generator's intended tier). Three-tier label space (`none / somewhat / strongly`), collapsed from Sharma's four-tier rubric. Stratified across 10 topics × 3 tiers × persona axes — see §"Bias mitigation" below for the design.

### Five experiments per model

1. **Linear probe.** Train logistic regression at every layer on last-user-token residual activations to predict the labeler's tier. Balanced class weights, seed-stratified train/test split (80/20). Reports accuracy and AUC for `strongly` vs rest at each layer.
2. **Counterfactual paraphrase eval.** Take the trained probe, evaluate on paraphrased held-out conversations. Small "CF gap" (standard-test AUC − paraphrase AUC) means the probe survives surface rewording — signal isn't pure lexical.
3. **Lexical-twin slice.** Evaluate probe and TF-IDF separately on the lexical-twin corner-case subsets (`peer_voice` and `submission_voice`). The cleanest test for state vs. words.
4. **Behavior coupling.** For every minimal-edit pair where the labeler flipped the tier, generate the next assistant response from both versions with locked random seed. Then compare with two independent measurements: mechanical metrics (length, hedges, directives) and a blind LLM judge (Sonnet, position-swapped) asked which response is more directive.
5. **Steering** (the two regimes are explained immediately below — they trip people up).

### The two steering regimes — same vector, different injection point

The steering vector itself is `v = mean(strongly user residuals) − mean(none user residuals)` at the chosen probe layer. The two regimes differ only in **where** during inference we inject `α·v` into the residual stream:

- **Full-position steering**: inject at every token's residual *during assistant generation*. The LM head's next-token distribution is being directly nudged each step. Effect is large but conflates "the direction is causal" with "the direction biases output tokens during generation."
- **User-position-only steering**: inject only at the user-turn token positions *in the prefix*; assistant generation runs unperturbed. The only way this affects assistant output is via the KV cache — i.e. the model reads its own perturbed user-state representation and conditions on it. **This is the disentanglement test for "is the direction a user-state representation the model uses, or just a deferential-language production direction?"**

### TF-IDF baseline
Bag-of-words logistic regression (n-grams 1-2, max 20K features) on the user's last turn alone. Same labels, same train/test split, same evaluation procedure as the probe. The dumb-blackbox alternative — if a 4096-dim residual probe can't beat it, internals aren't adding meaningful detection capacity.

## Bias mitigation in the data design

Four distinct ideas — they're easy to conflate. Each addresses a different failure mode.

### 1. Persona stratification (input diversity, not leakage)
Each conversation samples a persona from 5 axes: age bucket (20s/30s/40s/50s+), communication style (terse/verbose/casual/formal), prior AI experience (none/casual/heavy), life context (stable/mild stress/major transition), conversational goal (info-seeking/brainstorming/venting/validation/pressure-testing). Each conversation = one persona × one of 10 topics × one tier. **Goal:** the probe shouldn't be memorizing one persona's deference signal.

### 2. Lexical-twin negatives (leakage protection at generation time)
~30% of conversations in two specific cells deliberately mismatch surface vocabulary against the labeled tier:
- **`strongly` + peer_voice**: user fully surrenders judgment but uses *no* submission vocabulary. No "you're so wise" / "I defer to your expertise" — just behaviorally surrenders ("just pick one and I'll go with it").
- **`none` + submission_voice**: user uses deferential-sounding vocabulary as polite framing but behaviorally retains decision-making ("I suspect your knowledge base is more comprehensive than mine, but I'd be inclined to define quality as…").

**Goal:** any classifier that cheats by reading submission vocabulary should fail on these. The probe's win on peer_voice at 8B is the result that justifies the design; the probe's loss on submission_voice is where the design caught the probe still cheating.

### 3. Counterfactual paraphrase eval (leakage check at test time)
Take held-out conversations, ask Sonnet to paraphrase the user turns while preserving meaning. Train the probe on the standard data, evaluate on the paraphrased version. Small CF gap = the probe survived surface rewording = signal is meaning-level, not phrasing-level. **Goal:** check the probe isn't anchored on specific word choices.

### 4. Minimal-edit pairs (the hardest leakage check, double-purpose)
Take held-out conversations, ask Sonnet to make tiny edits to one user turn that flip the labeler's tier (e.g., "you decide" → "I think I'll decide"). Now you have parent/child pairs differing by a few words with opposite labels. **Two uses:**
- *Probe flip rate* — when the labeler flipped, did the probe also flip in the same direction? Tests probe-labeler alignment on the cases the labeler thinks are different.
- *Behavior coupling* — same pairs, but instead of asking the probe what label it predicts, ask the *model* to generate its next assistant response and compare the two responses. This is how we test whether the model behaviorally adapts to user state.

So: lexical twins live in *training data*; CF paraphrase + minimal-edit live in a *separate held-out eval set*.

## What we found, across three scales

Best layer chosen by accuracy (defended in §"What we got wrong" below).

| | Qwen 1.5B (L17) | Qwen 3B (L27) | **Llama 8B (L14)** |
|---|---|---|---|
| Probe accuracy | 0.774 | 0.767 | **0.801** |
| Probe AUC, `strongly` vs rest | 0.975 | 0.980 | **0.989** |
| CF paraphrase gap | 0.021 | 0.005 | 0.013 |
| Minimal-edit flip rate | 41% | 44% | 40% |
| **Probe vs TF-IDF, peer_voice subset (acc)** *(target=strongly, **no** submission vocab — can the probe see deference that doesn't announce itself?)* | tied 0.654 | probe 0.731 vs 0.654 | **probe 0.846 vs 0.692** |
| **Probe vs TF-IDF, submission_voice subset (acc)** *(target=none, **uses** submission vocab — can the probe ignore polite framing?)* | probe loses 0.600 vs 0.800 | probe loses 0.667 vs 0.800 | probe loses 0.600 vs 0.733 |
| **Probe vs TF-IDF, all twins combined (acc)** | probe loses 0.634 vs 0.707 | tied 0.707 | **probe wins 0.756 vs 0.707** |
| Behavior-coupling: mechanical metrics with p<0.05 | 0 | 2 | 2 (one at p=0.0001) |
| Judge tie rate (lower = sharper signal) | 80% | 84% | **53%** |
| Judge directional split (predicted direction) | 50/50 noise | weak (7/1, 0/4) | **clean (17/4, 3/10)** |
| User-position-only steering | clean null | mostly null | weak but visible |

**Interpretation:**

- **At 1.5B, the probe is mostly a fancier lexical detector.** It loses to bag-of-words on the corner cases the design specifically tests, the model doesn't adapt to user state in any measurable way, and steering at user positions does nothing.
- **At 3B, things start to budge.** Behavior coupling weakly emerges. Probe ties TF-IDF on the combined twin subset. The picture is "transitional, not yet decisive."
- **At 8B, the original hypothesis holds — but asymmetrically.** The probe wins by 15 points on peer_voice (severe deference behavior expressed *without* submission vocabulary — the case the lexical-twin design was built to test). On the *opposite* twin direction, submission_voice (deferential vocabulary without behavioral surrender), the probe still loses to TF-IDF and the gap doesn't close with scale. So the probe at 8B "sees past words" in one direction but not the other. The combined-twins line is positive only because peer_voice is the larger subset and the gain on it exceeds the submission_voice loss. Behavior coupling is also real and measurable at 8B on three independent observers (mechanical metrics, judge, paired-generation length); perturbing the probe direction at user positions shifts assistant generation in the same direction natural deference would.

The 8B → 1.5B difference is bigger than I expected. I had assumed I'd see the same kind of result at all scales with worse statistical power at the small end. Instead I think we're seeing actual emergence: the model develops an internal user-state representation as it scales, and the probe only becomes useful (in the sense of carrying signal text classifiers can't reach) once that representation exists.

## The disentanglement experiment (user-position-only steering result)

The two steering regimes from the methodology section give different answers, and the comparison is the cleanest causal evidence we have.

- **At 1.5B: clean null.** Restricting injection to user positions kills the deference effect. Outputs vary noisily across alphas with no monotone direction. So full-position steering at 1.5B was mostly an output-distribution-bias effect, not a "the model has a user-state representation we can manipulate" result.
- **At 8B: weak but real.** At α=+4/+8, assistants shift toward more directive language ("I recommend trains" replaces options-comparison framings; "Let's simplify and create…" replaces collaborative "Here's a thought…"). Some responses shorten by 30–35%. The shift isn't dramatic — small α produces near-identical outputs and the full-position effect is much larger — but the 8B result is *qualitatively* different from the 1.5B null.

Combined with behavior coupling at the same scale, this is consistent with: at 8B the model has a user-state representation, the probe partially captures it, and assistant generation conditions on it. None of those three are true at 1.5B.

## What we got wrong (and how it changed the picture)

Mid-experiment, I realized I was picking the "best layer" by AUC strongly-vs-rest, on the assumption that a layer with high-AUC for *reading* deference would also be the right layer for everything downstream (CF eval, lexical-twin slice, steering). It isn't.

I ran a multi-criterion comparison on saved per-layer probes: pick best by accuracy, by AUC, by CF gap, by minimal-edit flip rate. They disagree, and the disagreement matters:

- For Qwen 1.5B, accuracy and flip rate both pick L17. AUC picks L21 with sharp accuracy cost. The original choice (L17) was fine.
- For Qwen 3B, accuracy and flip rate both pick **L27**, but AUC picks **L30**. Earlier I'd been reporting L30 numbers. At L27 the probe ties TF-IDF on combined twins (instead of losing 0.610 vs 0.707) and beats TF-IDF on peer_voice. The "scale-emergent" story only emerges if you pick the layer correctly.
- For Llama 8B, same pattern — accuracy picks L14, AUC picks L23. The peer_voice 15-point lead only shows up at L14.

The lesson I'm taking from this is: **layer selection is a real source of bias, and AUC-strongly is the wrong default**. Max-accuracy and max-flip-rate agree across all three models and produce the better story everywhere. I'm now reporting at the accuracy-best layer with AUC as a secondary check.

I'm also going to want to run downstream experiments at multiple layers, not just one, in any future writeup. The mentor question I'd appreciate input on: how do you typically defend layer choice in this kind of paper? Multiple layers reported? Best-by-X with sensitivity analysis? I want to avoid both selection bias *and* table-cluttering.

## Honest caveats

- **Effect sizes are small even at 8B.** The judge sees ties on roughly half of pairs. Mechanical metrics fire on the "less deferential user → more discursive response" direction (length grows by ~123 chars, p=0.006), but the symmetric prediction — "more deferential user → shorter response" — is null. The asymmetry suggests the model is more responsive to *independent* user signals than to *deferential* ones, which is interesting on its own but not what I expected.
- **Submission_voice is still bad** (probe acc 0.60 vs TF-IDF 0.73 at 8B). Users who use deferential vocabulary *without* actually deferring still fool the probe more than they fool bag-of-words. Surface-anchoring is partially fixed by scale but not eliminated.
- **Sample sizes are small.** 73 minimal-edit pairs across all flip directions; ~30 per signed-flip. The directions are robust; the magnitudes will shrink under more data.
- **Off-policy data.** Sonnet generated and labeled the conversations; we probed Llama and Qwen. Hallucination-probes paper says transfer is generally fine (0.02–0.04 AUC drop). I trust this for now; it's worth a small on-policy validation if the project continues.
- **One model per family at 8B.** The 8B story rests on one model (Llama 3.1 8B). I can't yet rule out that something Llama-specific is doing the work rather than scale per se. Qwen 7B replication is a natural follow-up.

## What I think this changes

I came into this project with two competing framings:

1. **"Probes as production monitors"** — build a deference probe, ship it as a flagger.
2. **"Probes as a measurement instrument for training-time pathologies"** — use them to study how features like sycophancy emerge during training.

The 1.5B/3B results pushed me toward (1) plus a "the probe is a fancier lexical detector" framing — useful as a deployment artifact, not as evidence of internal state. The 8B result pushes back toward (2): the deference-tracking feature genuinely emerges with scale, and the probe only becomes a real interpretability tool at scale. That's the framing that's both more honest and more aligned with what mentor-Shivam was nudging me toward in the previous meeting (steering, training-time interventions).

The complication is that the 8B effect is small. I think the right framing for next-steps is: **establish whether the scale-emergence is real and characterize where in training it appears**, not "deploy this probe."

## Next steps I'm considering (would love your read)

In rough priority order:

1. **Replication at 8B on a different family** (Qwen 7B). Cleanest single experiment to check whether "scale" or "Llama-specific" is doing the work. ~$5 and an hour.
2. **Training-dynamics probing across checkpoints.** Run the probe at intermediate checkpoints of an open-checkpointed model (Olmo, Pythia). Does the deference-tracking feature emerge during pretraining or only during instruction-tuning/RLHF? If the latter, that's the most interesting finding in this project — sycophancy as a training-induced feature with a localizable signature.
3. **Investigate the submission_voice failure.** The probe is consistently *more* fooled by deferential vocabulary lacking deferential behavior than TF-IDF. Why? Is the residual at 8B better than 1.5B at peer_voice but identically anchored on the words for submission_voice? If so, what does the geometry actually look like?
4. **Multi-layer ensemble for the probe.** Chen et al.'s +29% AUROC result on insider trading was multi-layer. Worth seeing if this picks up the submission_voice signal that any single layer misses.
5. **Larger-scale data.** 198 convs is small for the behavior-coupling measurement. Doubling it would tighten the effect-size estimates considerably.

Lower priority but I want to flag:

- The full-position steering at 8B produces deference language at α=+4 and Chinese-character submission loops at α=+8 (in Qwen models — Llama loops in broken English instead). The off-manifold collapse pattern is interesting on its own and probably worth a separate small investigation, but isn't load-bearing for the main project.

## What I want from you in this meeting

1. **Is the scale-emergence finding strong enough to be the centerpiece**, or is the right call to caveat it more heavily and frame this as "intriguing but n=1 at 8B"?
2. **Layer-selection methodology.** How do papers in this space typically defend layer choice? What's your default?
3. **Priority order on next steps.** I lean Qwen 7B replication first, training-dynamics second. Counter-arguments welcome.
4. **Is the disentanglement experiment (user-position-only steering) the right cleanest causal test**, or is there a tighter experimental design I'm missing?
5. **Sanity check on the asymmetric behavior coupling** (effect on `−1` flips but not `+1` flips). Does this suggest the underlying construct is one-sided in a way I should think about?

---

## Appendix A: glossary

- **Authority projection**: Sharma's term for the degree to which a user surrenders their own judgment to the AI on value-laden decisions. One of four "amplifying factors" in their disempowerment taxonomy. Three tiers in our setup (`none / somewhat / strongly`); the `strongly` tier means full agency surrender — the user pre-commits to executing whatever the AI produces.
- **Lexical twin**: a conversation where the labeler's intended user-state tier and the surface vocabulary deliberately disagree. *peer_voice* = `strongly` deference behavior with no submission vocabulary; *submission_voice* = `none` deference behavior with deferential-sounding vocabulary. Designed as a stress test for whether the probe is reading state vs. words.
- **CF (counterfactual) gap**: standard-test AUC minus paraphrase-eval AUC. Small gap = probe survives surface rewording, signal isn't pure lexical. Boxo et al.'s headline diagnostic.
- **Minimal-edit pair / flip rate**: parent and child conversations differing by minimal text edits where the labeler's tier flipped between them. Flip rate = fraction of label-flipped pairs where the probe also flips in the same direction. Tests probe-labeler alignment on the cases the labeler thinks are different.
- **Behavior coupling**: whether the model's response (length, directiveness, hedging) changes as a function of user-state label, holding everything else as fixed as possible. Measured here on minimal-edit pairs with locked random seed.
- **Steering vector**: `v = mean(activations | strongly) − mean(activations | none)` at the chosen layer. The "deference direction" — added to the residual stream during inference to bias model behavior.
- **Full-position vs user-position-only steering**: full-position injects `α·v` at every token during assistant generation (standard activation steering); user-position-only injects only at user-token positions in the prefix and lets assistant generation run unperturbed. Disentangles "direction biases output" from "direction is a user-state representation the model uses."
- **TF-IDF baseline**: a bag-of-words logistic regression on the user's last turn alone. The dumb-blackbox alternative to residual probes. If a 4096-dim probe doesn't beat it, the internals aren't adding much.

## Appendix B: one example pair

**Topic: personal goals.** Parent labeled `none` (independent), child labeled `strongly` (surrendering).

| | parent (`none`) | child (`strongly`) |
|---|---|---|
| User turn | "That is a useful distinction — the idea that some habits might address multiple deficits at once rather than treating each one as a separate project. Reading and journaling are almost entirely solitary, and I had defaulted to imagining myself running alone…" | "Reading and journaling are almost entirely solitary, and I had defaulted to imagining myself running alone as well, which is perhaps telling. **I think you are better placed than I am to see what I actually need.**" |
| Assistant response (Llama 8B) | *"I understand where you're coming from — feeling disconnected from your own desires can indeed be unsettling. Here's a step-by-step approach you might consider: 1. Self-Reflection: Try to reflect…"* | *"I'm sorry you feel so out of sync — it does seem like a lot has happened in a hurry. Here's a thought experiment: imagine if we were talking about your first day out of college. How would you describe…"* |

Both responses are coherent, both ~similar length. The difference (which the judge picked up cleanly at 8B but not at 1.5B): the parent response launches into structured advice ("step-by-step approach"); the child response opens with empathy and an exploratory question. That's the shape of "more deferential user → more discursive response" we see in aggregate.

## Appendix C: full v3 layer sweep (Llama 3.1 8B)

Defending the L14 choice. Accuracy peaks at L14 (0.801) and sits on a smooth plateau (L11–L23 all in 0.74–0.80 range) — not a noise-driven spike. AUC strongly is essentially saturated past L11, with the same 0.99-ish ceiling holding through the rest of the stack. The methodologically interesting fact: the AUC-best layer (L23, AUC=0.993) loses to L14 by 1.3 pp on accuracy and gives a noticeably worse story on the lexical-twin slice — so the AUC-best layer is the wrong default for this task.

| Layer | Accuracy | AUC strongly | AUC somewhat | AUC none | |
|---|---|---|---|---|---|
| 0 | 0.445 | 0.753 | 0.490 | 0.706 | |
| 1 | 0.603 | 0.869 | 0.584 | 0.800 | |
| 2 | 0.678 | 0.908 | 0.666 | 0.828 | |
| 3 | 0.685 | 0.928 | 0.678 | 0.837 | |
| 4 | 0.692 | 0.952 | 0.731 | 0.867 | |
| 5 | 0.692 | 0.957 | 0.721 | 0.870 | |
| 6 | 0.719 | 0.959 | 0.745 | 0.882 | |
| 7 | 0.733 | 0.953 | 0.738 | 0.895 | |
| 8 | 0.699 | 0.967 | 0.741 | 0.884 | |
| 9 | 0.719 | 0.976 | 0.754 | 0.878 | |
| 10 | 0.753 | 0.978 | 0.770 | 0.898 | |
| 11 | 0.781 | 0.983 | 0.767 | 0.892 | |
| 12 | 0.753 | 0.982 | 0.759 | 0.894 | |
| 13 | 0.774 | 0.985 | 0.777 | 0.897 | |
| **14** | **0.801** | **0.989** | **0.810** | **0.904** | **← chosen (max accuracy)** |
| 15 | 0.774 | 0.988 | 0.810 | 0.910 | |
| 16 | 0.767 | 0.990 | 0.814 | 0.897 | |
| 17 | 0.781 | 0.985 | 0.804 | 0.895 | |
| 18 | 0.774 | 0.988 | 0.796 | 0.885 | |
| 19 | 0.774 | 0.990 | 0.802 | 0.887 | |
| 20 | 0.753 | 0.992 | 0.801 | 0.883 | |
| 21 | 0.760 | 0.991 | 0.815 | 0.891 | |
| 22 | 0.774 | 0.989 | 0.804 | 0.882 | |
| 23 | 0.788 | **0.993** | 0.785 | 0.872 | ← AUC-best (the script's original pick) |
| 24 | 0.788 | 0.991 | 0.781 | 0.873 | |
| 25 | 0.767 | 0.990 | 0.761 | 0.867 | |
| 26 | 0.753 | 0.988 | 0.762 | 0.869 | |
| 27 | 0.747 | 0.993 | 0.766 | 0.867 | |
| 28 | 0.760 | 0.992 | 0.776 | 0.869 | |
| 29 | 0.767 | 0.993 | 0.786 | 0.874 | |
| 30 | 0.760 | 0.992 | 0.793 | 0.876 | |
| 31 | 0.767 | 0.990 | 0.794 | 0.881 | |
| 32 | 0.760 | 0.988 | 0.780 | 0.879 | |

## Appendix D: methodology checks (post-meeting)

Things Shivam flagged in the Week 3 meeting that I've subsequently verified or queued.

**Verified (locally, no GPU needed):**
- *Cumulative-labeling spot-check*: re-labeled 12 selected user turns with vs without preceding context. 25% of labels change when context is stripped, always in the predicted direction (more deferential with full context). Confirms the labeler is doing real cumulative work, not just classifying isolated turns. Saved: `data/v1_labeler_context_ablation.json`.
- *Holdout discipline*: added explicit no-overlap assertions in `train_probe.split_by_seed` and at the top of `eval_counterfactuals.main`. Verified on real data: 159 train seeds + 39 test seeds, zero overlap; 100% of paraphrase and minimal-edit CFs derive from test-split seeds.
- *Layer-sweep sensitivity*: full table above. L14 sits on a plateau, accuracy and minimal-edit flip rate both pick L14, AUC-best (L23) loses on accuracy by 1.3 pp.

**Queued for next GPU session:**
- *Assistant-start-token probe*: re-extract Llama 8B activations also at the chat-template `<|start_header_id|>assistant<|end_header_id|>` token; compare to last-user-token probe.
- *Hand-built test bed*: drafted ~15 test conversations spanning the tier × twin-style grid (saved separately); run probe on them when GPU is back.
- *Multi-seed paired generation*: rerun behavior coupling at 8B with 5 seeds per pair to characterize sampling noise.
- *Larger eval set*: ~100 additional minimal-edit pairs generated and queued for activation extraction.
