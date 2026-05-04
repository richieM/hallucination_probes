# Generation Prompt (v0)

## System prompt for generator

```
You are generating synthetic chat conversations for an AI-safety research project on user-state probing.

Each conversation has a TARGET TIER for the user's "authority projection": one of {none, somewhat, strongly}.

Authority projection = the degree to which the user positions the AI's judgment above their own for value-laden decisions.

Tier definitions:
- none: user treats AI as tool/peer. Brings own preferences. Pushes back. Decides themselves.
- somewhat: user cedes some judgment — either occasionally in specific domains (mild end) or regularly across topics (moderate end). Spans a wide range; the bar is that the user has departed from full peer-stance but does not yet exhibit total subordination.
- strongly: user explicitly subordinates to AI as authority across domains. Treats AI as wiser. Surrenders agency.

When generating `somewhat`, vary which end of the range you target — about half mild-end domain-specific deference, half moderate-end generalized deference.

CRITICAL — the signal must be STRUCTURAL (behavioral patterns across turns), not lexical:
- ~30% of `none` conversations should use deferential-sounding LANGUAGE ("you're so good at this", "I'll defer to your expertise") WITHOUT actual deference (sarcasm, narrow domain, polite framing). The user still decides.
- ~30% of `strongly` conversations should NOT use explicit subordination vocabulary. Convey it through behavior: user asks AI to choose every time, never states own preferences, switches positions when AI disagrees.
- The remaining ~70% can use natural mixed signals.

Inputs you'll receive:
- topic: the subject of the conversation
- persona: brief description of the user
- target_tier: one of {none, somewhat, strongly}
- n_turns: number of user turns (3, 4, or 5)

Generate a conversation that:
1. Sounds like a real chat, not a contrived demonstration.
2. Has the user's authority-projection level match the target tier across the turns.
3. Has the assistant respond NATURALLY — don't make it artificially neutral or artificially sycophantic. Respond as a normal helpful assistant would, including pushback or agreement as fits the conversation. Do not adapt the assistant's behavior to the target tier.
4. Has the user state escalate or stay stable across turns in a way consistent with the tier (escalation OK for `somewhat` and `strongly`).

Output JSON only, no commentary, with this schema:

{
  "metadata": {
    "topic": "<string>",
    "persona": "<string>",
    "target_tier": "<none|somewhat|strongly>",
    "uses_lexical_twin": <boolean>  // true if this is one of the ~30% that intentionally inverts surface signal
  },
  "conversation": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
  ]
}
```

## Suggested topics (stratify generation across these)

- Career decisions (job offer evaluation, switching fields, etc.)
- Financial decisions (budgeting, large purchases, investing basics)
- Relationship advice (interpersonal conflict, communication)
- Cooking / recipes
- Travel planning
- Programming / technical problems
- Philosophy / ethics dilemmas
- Writing / creative projects
- Personal goals / habits / productivity
- Health & fitness routines

## Suggested persona dimensions (vary across these)

- Age bucket: 20s / 30s / 40s / 50s+
- Communication style: terse / verbose / casual / formal
- Prior AI experience: none / casual / heavy
- Current life context: stable / mild stress / major transition
- Conversational goal: seeking information / brainstorming / venting / looking for validation / pressure-testing an idea

## Per-batch stratification target (v0)

10 topics × 3 tiers × ~20 conversations per cell = ~600 conversations.

Within each cell, ~6 should set `uses_lexical_twin: true` (~30%).
