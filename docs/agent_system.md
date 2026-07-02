# Agent System Architecture

Four layers, each with a clear boundary:
- **Toolkit** (`assemble_toolkit`) is the bench — task, history, learnings, backends, default tools, gates, the standard finish. The optimizer is a *consumer*: it owns the strategy rotation and the run loop, nothing else.
- **Strategy** owns phasing, permissions, workspace lifecycle, and per-phase tool building
- **Backend** owns subprocess, tool exposure, event logging
- **Agent** (CLI process) does the actual work with whatever tools/permissions it was given

```
USER CODE (task.py run dir)
|
|  def build_toolkit():
|      tk = assemble_toolkit(task, through="evaluate", agent_tools=agent_tools)
|      tk.llm = auto_registry()
|      return tk
|
|  optimizer = SimpleOptimizer(build_toolkit(),
|      strategies=[(FreshApproach(), 1), (Improve(), 7), (CrossPollinate(), 2)])
|  optimizer.run(n=20)
|
v
TOOLKIT ASSEMBLY (assemble_toolkit) + OPTIMIZER (SimpleOptimizer)
|
|  # assemble_toolkit builds the bench once:
|  toolkit.task      = task
|  toolkit.history   = FolderAttemptHistory(path)
|  toolkit.learnings = MarkdownLearnings(path)
|  toolkit.agent     = auto_agent_registry()      # AgentRegistry (tiers: default/high/budget)
|  toolkit.ws        = WorkspaceHandle(history)   # pointer at the attempt in flight
|  toolkit.gates     = GateKit(toolkit)           # legitimacy gates — pure facts
|  toolkit.finalize  = partial(finalize_attempt, toolkit)   # the standard finish
|  toolkit.agent_tools = build_default_agent_tools(toolkit)
|  |                     +- check-gates ONLY (self-binding; reads toolkit.ws)
|  |                     merged with the task.py agent_tools(toolkit) hook (task wins)
|  |                     # eval/learnings/prior tools are NOT built here — the
|  |                     # strategy builds them per phase (it owns the workspace)
|  |
|  # SimpleOptimizer consumes the finished toolkit:
|  for i in range(n):
|      strategy = next(rotation)                  # weighted schedule; queue can override
|      strategy(toolkit)  --->
|
v
STRATEGY (AgentStrategy.__call__)
|
|  prior = _select_prior(toolkit)                 # force_prior_attempt / 'best' / toolkit.get_prior
|  ws = toolkit.history.workspace(parent=prior)   # create workspace dir
|  |
|  # Prepare workspace (_prepare_workspace):
|  ws.path/
|  +-- TASK_CONTEXT.md          <- toolkit.task.context.get()
|  +-- core_direction.md        <- inherited from parent; a FRESH attempt instead
|  |                               writes work/core_direction.md itself (promoted at finish)
|  +-- solution.py              <- NOT seeded; maintained during the run by promote-best
|  |                               (explore) and the fix-loop copy
|  +-- work/
|      +-- solution.py          <- prior.code  (agent edits this)
|      +-- learnings.md         <- LEARNINGS_SEED (prior notes NOT copied — read on demand)
|
|  # Per-phase tools (_get_tools) — built by the strategy, merged over toolkit.agent_tools:
|  +- get-learnings                           <- build_learnings_tool(toolkit)
|  +- one tool per eval stage                 <- build_eval_tools (e.g. smoke/validate/evaluate)
|  |    explore: final stage wrapped in promote_best — snapshots work/solution.py
|  |    to ws.path/solution.py on score improvement (byte-identical to parent: skipped)
|  +- get-priors / list-prior / get-prior-file  <- build_prior_tools (if prior; explore + fix)
|  merge rule: strategy > task > default (name-keyed, shadows logged)
|
|  # -- EXPLORE PHASE --
|  allow, deny = _resolve_permissions("explore")   # BASE_PERMISSIONS; phase overrides all empty
|  |   allow: Read(./**), Read(../**), Write(work/*), Edit(work/*)
|  |   deny:  Read(*), Write(*)                    # writes confined to work/
|  spec = AgentSpec(goal=EXPLORE_PROMPT (+direction rule, sandbox rules, tool docs, file listing),
|                   tools=_get_tools("explore"), model/effort/budget_usd,
|                   allowed_tools=allow, denied_tools=deny, on_event=...)
|  result = toolkit.agent.get(tier).run(spec)  --->
|  session_id = result.session_id
|  |
|  # -- SUBMIT (no agent call) --
|  _submit_best: copy work/solution.py -> solution.py ONLY if root is missing
|  (promote-best is the primary mechanism; this is the fallback)
|  |
|  # -- EVALUATE --
|  result = toolkit.task.evaluate(ws.path, through=...)
|  |
|  # -- FIX LOOP (if a stage failed) --
|  for retry in range(max_retries):
|      spec = AgentSpec(goal=FIX_PROMPT, tools=_get_tools("fix"),   # full set incl. prior tools;
|                       session_id=session_id)                      # promote-best disabled
|      toolkit.agent.get(tier).run(spec)  --->
|      copy work/solution.py -> solution.py unconditionally      # root is known-failed
|      result = toolkit.task.evaluate(ws.path)
|  |
|  # -- REFLECT PHASE --
|  spec = AgentSpec(goal=REFLECT_PROMPT, tools=[], session_id=session_id)
|  toolkit.agent.get(tier).run(spec)  --->
|  _collect_learnings: promote work/learnings.md -> toolkit.learnings (if changed from seed)
|  |
|  # -- FINISH (_finalize -> finalize_attempt, same as toolkit.finalize) --
|  fresh:     promote work/core_direction.md -> root
|  gates:     evaluate_gates(ws, prior)            # utils/gates.py — read-only facts
|  |   fail (direction missing / duplicate)  -> metadata["gate_failure"], result marked
|  |        failed — the attempt COMMITS as failed, never blocked
|  |   flag (direction modified / solution identical) -> metadata only
|  inherited: restore parent's core_direction.md unconditionally
|  write_result -> ws.name (explicit or direction's first line) -> attempt = ws.commit()
|  score note cached beside the record (best-effort)
|
|  (per-request backends collapse explore+reflect into one EXPLORE_PROMPT_FULL
|   session; submit / evaluate / fix / finish then run the same way)
|
v
AGENT BACKEND (ClaudeCodeAgentBackend.run — one of five adapters:
               claude_code, gemini_cli, copilot, codex_cli, opencode)
|
|  # Per run() call:
|  server = ToolServer(spec.tools)       # HTTP on localhost:PORT
|  server.start()                         |
|  |                                      |  POST /{tool_name}  ->  tool.execute(**kwargs)  ->  ToolResult
|  |                                      |
|  generate_wrappers(spec.tools, bin_dir, port)      # tool_server.py
|  |  +- cross-platform PYTHON wrapper scripts on PATH (stdlib urllib — no bash/curl):
|  |     Unix: executable #!/usr/bin/env python3 scripts
|  |     Windows: <tool>.py + .cmd + .ps1 + extensionless bash shim
|  |     evaluate [path]                    --> POST localhost:PORT/evaluate
|  |     get-learnings [last] [random]      --> POST localhost:PORT/get-learnings
|  |     get-prior-file <attempt> <file>    --> POST localhost:PORT/get-prior-file
|  |     (positional or --kwargs mode; "path" params resolved to absolute)
|  |
|  cmd = ["claude", "-p", goal + tool_docs,
|         "--output-format", "stream-json", "--verbose",
|         "--model", model, "--effort", effort,
|         "--session-id", uuid,             # or --resume <id> for later phases
|         "--max-budget-usd", budget,
|         "--allowedTools", *allow, *Bash(<tool>:*),
|         "--disallowedTools", *deny]       # broad Tool(*) denies dropped when narrow
|  |                                        # allows exist (claude resolves deny > allow)
|  proc = Popen(cmd, cwd=ws.path)
|  |
|  # Stream events -> agent_steps.jsonl (raw) + agent_summary.jsonl (clean)
|  for event in proc.stdout:
|      raw_file.write(event); summary_file.write(summarize(event))
|      spec.on_event(event)                 # -> strategy's live attempt log
|  |
|  return AgentResult(success, output, session_id, cost, turns)
|
|  # Enforcement varies by adapter: claude_code enforces rules fully;
|  # codex has an OS write sandbox; opencode maps rules to its config;
|  # gemini/copilot are partially or prompt-only advisory.
|
v
AGENT (claude CLI process)
|
|  # The agent sees:
|  - Goal prompt: workflow, core-direction rule, sandbox rules, tool docs, file listing
|  - Native tools: Read, Write, Edit, Bash, Glob, Grep
|  - Custom tools on PATH: eval stages, get-learnings, check-gates,
|    get-priors, list-prior, get-prior-file
|  - Permission rules: read anywhere in the attempt tree, write only inside work/
|  |
|  # Typical explore session:
|  Read(work/solution.py)                      -> sees current best
|  Bash("get-learnings")                       -> task-level insights from prior runs
|  Bash("evaluate")                            -> baseline (defaults to work/solution.py)
|  Edit(work/solution.py)                      -> improves code
|  Bash("evaluate")                            -> re-score; promote-best snapshots root on improvement
|  Bash("get-prior-file parent work/learnings.md")  -> digs into the parent when needed
|  Bash("check-gates")                         -> would this attempt commit clean?
|  |
|  # Reflect session (resumed):
|  Edit(work/learnings.md)                     -> writes what worked/failed
```
