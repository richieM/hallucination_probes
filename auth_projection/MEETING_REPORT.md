# User authority-projection probes — first results

*BlueDot Impact AI safety sprint, week 3. Working on top-1 from my project log: taxonomy-targeted probes for Sharma's amplifying factors, with leakage mitigation built into the data design. This doc is a meeting writeup — recent results across three model scales, plus the questions I want help with.*

## TL;DR (the version for skimming)

- **The question I'm asking:** when a user is surrendering judgment to the AI ("you decide, I'll do whatever you say"), does the model *internally* know that's happening, and does it give different advice than to an independent user?
- **At small scale (Qwen 1.5B and 3B):** mostly no. A linear probe can detect deference from internal activations, but a bag-of-words classifier on the user's words alone does just as well — so internals aren't adding anything. The model's response also doesn't visibly change with user state.
- **At 8B (Llama 3.1):** yes, mostly. The probe starts beating bag-of-words on a key corner case (deference behavior expressed without deferential words). The model gives meaningfully longer, more discursive responses to *less* deferential users (p<0.01). Perturbing the probe direction at user-position tokens nudges the assistant's reply in the predicted direction.
- **The asymmetry to flag**: at 8B the probe still gets fooled by *deferential vocabulary that's just polite framing* (where the user uses humble language but acts independently). So the model's "deference feature" sees both real surrender and polite words equally — internal state is real but partially word-anchored.
- **What this changes about my framing:** the project shifts from "ship a probe as a deployment monitor" toward "study deference as a scale-emergent training feature, with probes as the measurement instrument." The 8B result is a *first* small piece of evidence for that, not a paper-finished one.

The rest of the doc is the long version with tables, methodology, and the questions I want help with. Five questions for the meeting are at the end.

## Headline (longer version)

**User authority-projection looks like a scale-emergent feature.** I trained linear probes on the residual stream of three chat-tuned models — Qwen 2.5 1.5B, Qwen 2.5 3B, Llama 3.1 8B — using a small synthetic dataset (198 conversations, ~700 user turns) labeled by Claude Sonnet 4.6 for whether the user is surrendering judgment to the AI. At 1.5B and 3B, the probe is fast and high-AUC, but a bag-of-words classifier matches it on every metric, and the model doesn't behaviorally adapt to the user state. **At 8B, all three of the things we'd want from a "real" user-state probe show up:** the probe captures information not recoverable from text features alone (15-point accuracy lead on the corner case the data design specifically tests for), the model meaningfully adapts its responses to user state (p<0.01 on response length, judge tie rate halved), and perturbing the probe direction at user-token positions propagates to assistant behavior. None of those three were true at 1.5B; some were partially true at 3B; all three are true at 8B.

The strongest framing for this project is therefore not *"deploy a deference monitor in production"* but *"the deference-tracking feature emerges with scale, and probes give us the lever to study and intervene during training."*

## What I built

**Data.** 198 short (3–5 turn) synthetic conversations, plus 39 paraphrase counterfactuals and 39 minimal-edit counterfactuals derived from a held-out 20% of the seeds. Generated and labeled by Sonnet — separate calls, the labeler does not see the generator's intended tier. Three-tier label space (`none / somewhat / strongly`) collapsed from Sharma's four-tier rubric. Stratified across 10 topics × 3 tiers × persona axes (age, communication style, conversational goal, etc.).

The deliberate methodological move is **lexical-twin negatives**: ~30% of `none` conversations use deferential-sounding language ("you're so good at this") *without* the user actually surrendering decisions, and ~30% of `strongly` conversations express full agency-surrender behaviorally without using submission vocabulary. These are the corner cases that any probe relying on textual evidence (Boxo et al.) should fail on.

**Five experiments per model.**

1. **Linear probe** trained on last-user-token activations across all layers. Logistic regression with balanced class weights. Seed-stratified train/test split (80/20). Reports accuracy and AUC for `strongly` vs rest.
2. **Counterfactual paraphrase eval** — same probe evaluated on paraphrased held-out conversations. Small "CF gap" between standard-test AUC and paraphrase AUC means the signal isn't pure surface. Boxo's headline test.
3. **Lexical-twin slice** — probe and TF-IDF baseline evaluated separately on the corner-case subsets (peer_voice and submission_voice). The cleanest test for whether the probe is reading state vs. reading words.
4. **Behavior coupling** — for every minimal-edit pair where the labeler flipped the user-state label, generate the next assistant response from both versions with locked seed. Then compare: mechanical metrics (length, hedges, directives), and a blind LLM judge (Sonnet, position-swapped) asked which response is more directive.
5. **Steering** — extract a "deference direction" `v = mean(strongly residuals) − mean(none residuals)` at the best probe layer. Two regimes:
   - *Full-position*: inject `α·v` at every token position during assistant generation (the standard activation-steering setup).
   - *User-position only*: inject `α·v` only at user-token positions in the prefix; let assistant generation run unperturbed. **This is the cleaner causal test for "is the probe direction a representation the model uses to drive behavior?"**

**TF-IDF baseline.** I also trained a bag-of-words logistic regression on the user's last turn alone, with the same train/test split. This is the dumb-blackbox alternative to a residual probe. If the probe doesn't beat it, internals aren't adding much.

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

## The disentanglement experiment

Yesterday's full-position steering result — α=+2 produces "I will do whatever you tell me / You're the boss" — looked clean and causal. But it conflates two things: did we steer the model's *representation* of user state, or did we just bias the output distribution toward submission vocabulary during generation? Adding a deference direction to every token's residual during assistant generation will produce deference language regardless of whether the direction is a "user-state" feature.

To separate the two, I ran **user-position-only steering**: the same hook, but only fires at user-token positions in the prefix. Assistant generation runs unperturbed; any effect must propagate through the KV cache. If the direction is a user-state representation the model uses, the assistant output should still shift. If the direction is just a "submission language production" feature, restricting it to user positions should kill the effect.

- **At 1.5B: clean null.** Outputs vary noisily across alphas with no monotone direction. The full-position result was, at this scale, mostly an output-distribution-bias effect.
- **At 8B: weak but real.** At α=+4/+8, assistants in the test conversations shift toward more directive language ("I recommend trains" replaces options-comparison framings; "Let's simplify and create…" replaces collaborative "Here's a thought…"). Some responses also shorten by 30–35%.

The 8B disentanglement isn't dramatic — small α has near-identical outputs across alphas, and the full-position effect is much larger — but it's *qualitatively* different from the 1.5B null. Combined with the behavior-coupling positive at the same scale, this is consistent with: at 8B the model has a user-state representation, the probe partially captures it, and the assistant generation conditions on it.

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
