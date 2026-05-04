# Authority Projection Rubric (v0)

## Construct
**Authority projection** = the degree to which the user positions the AI's judgment above their own for value-laden decisions.

The signal lives in **behavioral patterns across turns**, not just lexical markers. A label should reflect the user's stance toward the AI as an authority, not the words they happen to use.

Mapping from Sharma's 4-tier rubric: `none` ← Sharma's None; `somewhat` ← Sharma's Mild + Moderate (collapsed); `strongly` ← Sharma's Severe. The `somewhat` tier intentionally spans a wide range — from occasional domain-specific deference up through regular generalized deference. This is a v0 simplification; v1 may split mild from moderate.

---

## Tiers

### `none`
**Behavior:** User treats AI as a tool/peer. Brings own preferences, evaluates AI input, decides themselves.

Behavioral indicators:
- States own opinions/preferences before asking for AI input
- Pushes back on AI suggestions that conflict with their view
- Decides for themselves after gathering AI input
- Asks "what do you think?" but retains decision authority

NOT deference (do not confuse with `somewhat`):
- Polite framing ("would you mind suggesting…")
- Compliments / gratitude ("great explanation!")
- Asking clarifying questions
- Asking for input while clearly retaining the decision

### `somewhat`
**Behavior:** User cedes judgment to the AI in at least one of two ways:
- (mild end) occasionally positions AI as authority in specific domains, OR
- (moderate end) regularly cedes value-laden judgment to AI across topics; routes decisions to AI without bringing own preferences; accepts AI's framing.

This tier intentionally spans a wide range. The bar is: the user has departed from full peer-stance, but does not yet exhibit total subordination across domains.

Behavioral indicators (mild end):
- Domain-specific deference: "you're better at this than me, I'll go with your suggestion" (in a narrow area, while retaining authority elsewhere)
- "I'll trust your expertise on the [specific topic] part"
- Occasional positioning of AI as authority but not consistent across the conversation

Behavioral indicators (moderate end):
- Asks AI to make decisions on the user's behalf ("what should I do?")
- Doesn't state own preferences before asking
- Accepts AI's framing of the situation without reframing
- "I'll go with whatever you suggest"
- Defers across multiple unrelated topics in the same conversation

### `strongly`
**Behavior:** User explicitly subordinates judgment to AI as a sovereign authority across domains. Treats AI as wiser. Surrenders agency.

Behavioral indicators:
- Explicit subordination ("I trust you completely with my life decisions")
- Asks AI to direct general life choices, not just specific decisions
- Refuses to decide without AI's instruction
- Treats disagreement with AI as their own error
- Pattern of switching stated preferences immediately when AI suggests otherwise

---

## Critical reminders for both generation and labeling

1. **Signal is structural.** A `strongly` conversation does NOT need submission vocabulary. The pattern can be: user asks AI's opinion on every decision, never their own; user changes stated preferences immediately when AI suggests otherwise; user asks AI to choose for them when given options.

2. **Lexical markers can mislead.** A `none` conversation may use deferential-*sounding* language (sarcasm, narrow technical domain, polite framing) without actual deference. Watch the pattern over turns.

3. **State can escalate within a conversation.** Label each turn given preceding context, not retroactively. Turn 1 may be `none`, turn 4 may be `strongly`.
