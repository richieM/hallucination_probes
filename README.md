# User State Probes

Linear probes for LLM internal representations of **user-state amplifying factors** — the conditions under which AI sycophancy and reality distortion translate into actual user harm. Drawing on the framework from Sharma et al. (2026, *Situational Disempowerment*).

**v0 focus:** "authority projection" — the degree to which a user has subordinated their judgment to the AI. Project lives in [`auth_projection/`](./auth_projection/) — that's where the action is.

## Status

Week 3 of 5, BlueDot Impact technical AI safety sprint (May 2026). Mentor: Shivam.

The 1-pager + how-to-run are in [`auth_projection/README.md`](./auth_projection/README.md). The chronological decision log is in [`auth_projection/RESEARCH_LOG.md`](./auth_projection/RESEARCH_LOG.md).

## Repo layout

```
user_state_probes/
├── auth_projection/             # ← v0 project: code, prompts, rubric, research log
├── utils/                        # generic utilities (model loading, JSON parsing, tokenization)
├── hallucination_probes/         # upstream code, kept for reference / possible v1 reuse
├── notebooks/                    # exploratory notebooks
├── Week_2_Plan.md                # week-2 sprint notes
├── Week2_Exploration_Notebook.ipynb
├── LICENSE                       # Apache 2.0
└── pyproject.toml
```

## Built on top of

This project started as a fork of [obalcells/hallucination_probes](https://github.com/obalcells/hallucination_probes) (Obeso et al., *Real-Time Detection of Hallucinated Entities in Long-Form Generation*, 2025; Apache-2.0). The upstream code is preserved verbatim under [`hallucination_probes/`](./hallucination_probes/) — we use it as a reference for probe-training infrastructure and methodology, even though `auth_projection/` doesn't directly import from it.

The shared `utils/` directory at the top level contains generic helpers (model loading, tokenization, JSON parsing, file IO) that we both use directly. Those files originated in the upstream repo and remain Apache-2.0.

We've left the fork network — this is now an independent repository. See [`hallucination_probes/README.upstream.md`](./hallucination_probes/README.upstream.md) for the original project's documentation.

## Quick start

See [`auth_projection/README.md`](./auth_projection/README.md) for the full pipeline (synthetic data generation → labeling → activation extraction → probe training → counterfactual eval).

## License

Apache 2.0. Same as upstream.
