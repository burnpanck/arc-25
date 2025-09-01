
## CoT

We'll try to teach the agent to think in the following steps:
- Given a prompt, containing
  * the `<input>` sections,
  * some pre-computed `<facts>`, and
  * any `<reference>` (DSL cheat-sheet, examples retrieved by RAG, etc...)
  * plus a direct instruction to the LLM
- Hypothesize: Output a natural language `<hypothesis>` about
  the concepts could make up the rule (what is information, what is distraction),
  and point-out ambiguities.
  **Important:** This should only ever describe the input, never the output.
- Commit: Output a natural language `<rule>`, describing the underlying rule.
- Plan: Output a natural language or pseudo-code `<plan>` on how to implement the rule.
- Implement: Output an `<implementation>` in our DSL which implements the rule according to the plan.

In the first version, we'll let the model generate all of that in a single step.
