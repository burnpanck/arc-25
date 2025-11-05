
## CoT

We'll try to teach the agent to think in the following steps:
- Given a prompt, containing
  * the `<inputs>` sections,
  * some pre-computed `<facts>`, and
  * any `<reference>` (DSL cheat-sheet, examples retrieved by RAG, etc...)
  * plus a direct instruction to the LLM
- Hypothesize: Output a natural language `<input-descr>` / `<output-descr>` about
  the concepts could make up the rule (what is information, what is distraction),
  and point-out ambiguities.
  **Important:** `<input-descr>` should describe only the input images, but it may interpret
  the input objects in a way informed by the rule/output.
- Perhaps we should insert free-text considerations in-between to encourage stepwise reasoning.
- Commit: Output a natural language `<rule>`, describing the underlying rule.
- Plan: Output a natural language or pseudo-code `<plan>` on how to implement the rule.
- Implement: Output an `<implementation>` in our DSL which implements the rule according to the plan.

In the first version, we'll let the model generate all of that in a single step.


## Next steps
- **Telemetry**: log compile_rate, diffs, and final pass on training I/O.
- **Runner skeleton**: prompt → code → sandbox run → simple repair (1 pass).
- **RAG hookup**: index DSL docs + the example solutions; add 3–5 helpful chunks to the prompt.
  (ask ChatGPT for a tiny **FAISS/Chroma** indexer snippet for RAG.)
- **Top-k candidates** (k=2–3) + temp ladder on repairs.
