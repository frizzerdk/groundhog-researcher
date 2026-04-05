# Agent System Architecture

Four layers, each with a clear boundary:
- **Optimizer** builds the toolkit (tools, backends, capabilities)
- **Strategy** owns phasing, permissions, workspace lifecycle
- **Backend** owns subprocess, tool exposure, event logging
- **Agent** (CLI process) does the actual work with whatever tools/permissions it was given

```
USER CODE (task.py)
|
|  task = MNISTTask()
|  optimizer = SimpleOptimizer(task, strategies=[(AgentStrategy(budget_usd=0.50), 1)])
|  optimizer.toolkit.llm = auto_registry()
|  optimizer.run(n=5)
|
v
OPTIMIZER (SimpleOptimizer)
|
|  # Build toolkit once at init:
|  toolkit.task      = task
|  toolkit.history   = FolderAttemptHistory(path)
|  toolkit.learnings = MarkdownLearnings(path)
|  toolkit.llm       = auto_registry()          # BackendRegistry
|  toolkit.agent     = auto_agent_registry()     # AgentRegistry
|  toolkit.agent_tools = build_default_agent_tools(toolkit)
|  |                     +- get-learnings  <- agent_tool(func=toolkit.learnings.get, params={last, random})
|  |                     +- smoke          <- agent_tool(func=L path: stage.call(open(path).read()))
|  |                     +- validate       <- agent_tool(func=L path: stage.call(open(path).read()))
|  |                     +- evaluate       <- agent_tool(func=L path: stage.call(open(path).read()))
|  |
|  # Run loop:
|  for i in range(n):
|      strategy(toolkit)  --->
|
v
STRATEGY (AgentStrategy.__call__)
|
|  prior = toolkit.get_prior(toolkit)           # select attempt to improve
|  ws = toolkit.history.workspace(parent=prior)  # create workspace dir
|  |
|  # Prepare workspace:
|  ws.path/
|  +-- TASK_CONTEXT.md          <- toolkit.task.context.get()
|  +-- solution.py              <- prior.code  (READ-ONLY during explore)
|  +-- approach.md              <- from prior attempt
|  +-- learnings.md             <- LEARNINGS_SEED template
|  +-- workspace/
|      +-- temp_solution.py     <- prior.code  (agent edits this)
|
|  # -- EXPLORE PHASE --
|  tools  = _get_tools(toolkit, "explore")   # all toolkit.agent_tools
|  allow, deny = _resolve_permissions("explore")
|  |   allow: Read(*), Write(workspace/*), Write(learnings.md), Edit(workspace/*), Edit(learnings.md)
|  |   deny:  Write(*), Write(solution.py), Edit(solution.py), Write(TASK_CONTEXT.md), ...
|  |
|  spec = AgentSpec(
|      goal       = EXPLORE_PROMPT.format(eval_command="validate", ...),
|      workspace  = ws.path,
|      tools      = [get-learnings, smoke, validate, evaluate],
|      model/effort/budget_usd,
|      allowed_tools = allow,
|      denied_tools  = deny,
|  )
|  result = toolkit.agent.get("default").run(spec)  --->
|  session_id = result.session_id
|  |
|  # -- SUBMIT PHASE --
|  spec = AgentSpec(goal=SUBMIT_PROMPT, tools=[], session_id=session_id,
|                   allowed: +Write(solution.py), +Edit(solution.py))
|  toolkit.agent.get("default").run(spec)  --->
|  |
|  # -- EVALUATE --
|  result = toolkit.task.evaluate(ws.path, through="evaluate")
|  |
|  # -- FIX LOOP (if errors) --
|  for retry in range(max_retries):
|      spec = AgentSpec(goal=FIX_PROMPT, tools=[all], session_id=session_id)
|      toolkit.agent.get("default").run(spec)  --->
|      result = toolkit.task.evaluate(ws.path)
|  |
|  # -- REFLECT PHASE --
|  spec = AgentSpec(goal=REFLECT_PROMPT, tools=[], session_id=session_id,
|                   denied: +Write(solution.py))
|  toolkit.agent.get("default").run(spec)  --->
|  toolkit.learnings.add(ws.path / "learnings.md")
|  |
|  attempt = ws.commit(result, metadata={strategy, prior, cost})
|
v
AGENT BACKEND (ClaudeCodeAgentBackend.run)
|
|  # Per run() call:
|  server = ToolServer(spec.tools)       # HTTP on localhost:PORT
|  server.start()                         |
|  |                                      |  POST /{tool_name}  ->  tool.execute(**kwargs)  ->  ToolResult
|  |                                      |
|  generate_wrappers(spec.tools, bin_dir, port)
|  |  +- bash scripts on PATH:
|  |     get-learnings [last] [random]     --> curl POST localhost:PORT/get-learnings
|  |     smoke <path>                      --> curl POST localhost:PORT/smoke
|  |     validate <path>                   --> curl POST localhost:PORT/validate
|  |     evaluate <path>                   --> curl POST localhost:PORT/evaluate
|  |
|  cmd = ["claude", "-p", goal + tool_docs,
|         "--output-format", "stream-json",
|         "--model", model, "--effort", effort,
|         "--max-budget-usd", budget,
|         "--session-id", uuid,
|         "--allowedTools", *allow,
|         "--disallowedTools", *deny]
|  |
|  proc = Popen(cmd, cwd=ws.path)
|  |
|  # Stream events -> agent_steps.jsonl + agent_summary.jsonl
|  for event in proc.stdout:
|      raw_file.write(event)
|      summary_file.write(summarize(event))
|  |
|  return AgentResult(success, output, session_id, cost, turns)
|
v
AGENT (claude CLI process)
|
|  # The agent sees:
|  - Goal prompt with workflow instructions
|  - Native tools: Read, Write, Edit, Bash, Glob, Grep
|  - Custom bash tools: get-learnings, smoke, validate, evaluate
|  - Permission rules restricting file access
|  |
|  # Typical explore session:
|  Read(solution.py)                          -> sees current best
|  Bash("get-learnings")                      -> sees prior run insights
|  Edit(workspace/temp_solution.py)           -> improves code
|  Bash("validate workspace/temp_solution.py") -> gets StageResult back
|  Edit(workspace/temp_solution.py)           -> iterates
|  Bash("validate workspace/temp_solution.py") -> checks improvement
|  Bash("evaluate workspace/temp_solution.py") -> full eval
|  |
|  # Submit session (resumed):
|  Bash("cp workspace/temp_solution.py solution.py")
|  |
|  # Reflect session (resumed):
|  Edit(learnings.md)                         -> writes what worked/failed
```
