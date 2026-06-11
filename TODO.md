# TODO — code catching up to the vault

Design is anchored in the vault (FrizzerNotes/…/GroundhogResearcher);
this is the implementation backlog it implies.

- [x] Build `tools/attempt_logger.py` — polymorphic event logger
      (replaces `tools/conversation_log.py`, reshapes `tools/attempt_log.py`).
      Design: vault "Markdown Attempt Logger" + "Attempt Log".
- [x] Per-event cost: events carry `cost` + `usage`; estimator reads
      `attemptlog.jsonl`; all backend PRICING dicts wired into
      `tools/cost_estimate.py` defaults. Design: vault "Cost Tracking".
- [ ] Add `Task.get_agent_tools()` so tasks can register agent tools.
      Design: vault "Agent Tool Building".
- [ ] Add `build_promote_tool` — explicit agent-driven promotion.
      Design: vault "Agent Tool Building".
- [ ] Relocate `toolkit.get_prior` -> `toolkit.workspace.get_prior`.
      Design: vault "Toolkit Implementation".
- [ ] Retire the deprecated `per_request` agent path once no backend
      declares it. Design: vault "Agent Backend".
- [ ] Evaluate a container-based sandbox as a uniform enforcement floor
      under all agent backends. Design: vault "Agent Sandbox".
