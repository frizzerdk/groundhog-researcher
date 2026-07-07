# Agent Sandboxing

How groundhog confines what an agent can read, write, and execute — and
where the boundaries actually are per backend.

## Goal

Each agent should:

1. **Read** files in its attempt directory (`./` and `../` for sibling
   attempts and shared materials). System paths and unrelated project
   sources should be **denied**.
2. **Write/edit** files in `work/` only. Writes to the attempt root,
   sibling attempts, parent dirs, or system paths should be **denied**.
3. **Run only the registered Groundhog tools** (e.g., `evaluate`,
   `get-learnings`) and the agent CLI's built-in tools (Read/Edit/Bash).
   Free-form bash to circumvent the above should be **denied**.

These goals are encoded in `BASE_PERMISSIONS` in
`groundhog/strategies/agent.py` and translated per backend.

## Two layers of enforcement

### Layer 1 — explicit rules

Each backend translates `AgentSpec.allowed_tools` /
`AgentSpec.denied_tools` into its own permission system. Some translate
to OS-level enforcement, some to CLI-level filters, some to advisory
prompt text. See the matrix below for what's hard vs. soft.

### Layer 2 — prompt-level guidance

The strategy's explore/fix/reflect prompts spell out the rules in
human-readable form and explicitly tell the agent **not to circumvent
them**. This is necessary because:

- For backends where rules are advisory only (codex, gemini), the prompt
  is the only signal.
- For backends where rules are hard (claude_code), the prompt prevents
  the agent from wasting turns trying things that will be blocked.
- It documents the contract for human reviewers reading the transcript.

The rules section is in `EXPLORE_PROMPT` /
`EXPLORE_PROMPT_FULL` / `FIX_PROMPT` in `agent.py`.

## Per-backend reality

| backend | reads outside attempt | writes outside `work/` | bash gate | mechanism |
|---|---|---|---|---|
| **claude_code** | ✗ blocked | ✗ blocked | ✗ blocked | `--allowedTools` / `--disallowedTools` (full enforcement) |
| **opencode** | ✗ blocked | ✗ blocked | ✗ blocked | generated `opencode.json` permission config (absolute-path patterns for read/list/external_directory; workspace-relative for edit) |
| **gemini_cli** | ✗ blocked by built-in workspace isolation | ⚠ writes outside cwd blocked, but no work/-only granularity | ⚠ blocks via built-in `LocalAgentExecutor` for `run_shell_command` | gemini's workspace isolation + advisory prompt |
| **copilot** | ✓ via copilot's own safety | ⚠ work/ allowed but **attempt-root writes leak** (path-deny patterns are buggy upstream — see [copilot-cli #2722](https://github.com/github/copilot-cli/issues/2722)) | ⚠ blanket `shell` deny silently dropped | `--deny-tool` for path-specific only |
| **codex_cli** | ⚠ advisory only — model can ignore | ⚠ `work/` ✓ but **attempt-root + outside-attempt LEAK** on Windows | ⚠ advisory only | `-s workspace-write` + advisory prompt; OS-level boundary is Windows ACL-based and **cannot restrict beyond user write permissions** ([codex Windows analysis](https://codex.danielvaughan.com/2026/04/01/codex-cli-windows-native-sandbox-wsl/)) |

`✓` = enforced · `⚠` = partial / has known gap · `✗` = blocked

## Known limitations

### codex_cli on Windows

OpenAI's codex CLI uses an AppContainer-based sandbox on Windows. Per
their own docs:

> The primary limitation is that it cannot prevent writes to directories
> where the Everyone SID already has write permissions. The sandbox does
> not prevent file writes, deletions, or creations in any directory
> where the Everyone SID already has write permissions.

Since attempt directories live under the user profile (which the user
account owns), codex's restricted token still has write access to
sibling attempts and the parent directory. We've tried:

- `--skip-git-repo-check` — already used; doesn't change boundary.
- `-c project_root_markers=[]` — disables walk-up search for `.git`,
  but doesn't shrink the writable area.
- `-c sandbox_workspace_write.writable_roots=[]` — already empty; the
  default workspace (cwd) is broader than literal cwd.

Workarounds that *would* fix it:

- **WSL2 mode** — codex supports running under WSL with Linux Landlock
  isolation, which is strict (kernel-level). Trade: agent runs in
  Linux, paths translate, perf hit.
- **Docker wrapper** — see "Future: Docker isolation" below.

The AppContainer also cuts the *other* way: it cannot traverse
`WindowsApps` reparse points, so a venv based on the **Microsoft Store
Python** (`pyvenv.cfg` `home = ...WindowsApps...`) is unlaunchable from
inside the sandbox. Every generated tool wrapper then fails with
`No Python at "...WindowsApps..."` — the agent sees broken tools, not a
broken interpreter, and runs blind. `generate_wrappers` warns when it
detects this; the cure is rebasing the env on a regular interpreter
(`uv python install && uv python pin && uv sync`).

### copilot path-pattern denies

GitHub copilot-cli's `--deny-tool 'write(<path>)'` pattern matching is
broken upstream ([issue #2722](https://github.com/github/copilot-cli/issues/2722)).
A pattern targeting `read(...)` denies ALL reads regardless of path. We
work around this by skipping blanket denies entirely and accepting that
attempt-root writes leak. Will revisit when upstream is fixed.

### opencode model compliance

The constraints in the generated `opencode.json` are correctly enforced
(filesystem ground truth confirms — no probe writes escape). But the
sonnet/deepseek models, when given an "exploratory" prompt, sometimes
treat the workspace as background context and produce a summary instead
of executing the requested operations. This is a model-behavior issue,
not a constraint issue.

The production explore prompt is more directive ("Run X, then Y, then
Z") and the models follow it reliably; the issue mostly surfaces in the
synthetic probe.

## Future: Docker isolation

The strongest realistic isolation is to run each agent inside a Docker
container with the attempt directory bind-mounted. The container's
filesystem becomes the sandbox: writes outside the mount go to the
ephemeral container FS and are discarded; reads outside the mount fail
because they're not visible.

### Sketch

```python
class DockerSandboxedBackend(AgentBackend):
    """Wraps an inner backend so the CLI runs inside a docker container."""

    def __init__(self, inner: AgentBackend, image: str = "groundhog/agent-runtime"):
        self.inner = inner
        self.image = image

    def run(self, spec: AgentSpec) -> AgentResult:
        # Build the inner backend's command, but prefix with `docker run`.
        # Bind-mount workspace; share the tool server via host-gateway.
        ...
```

Usage:

```python
toolkit.agent.register("safe", DockerSandboxedBackend(ClaudeCodeAgentBackend()))
# Then cfg.tier = "safe" picks the docker-wrapped variant.
```

### Costs

- **System-level dependency**: Docker Engine / Docker Desktop must be
  installed and running. Not a Python `pip` dependency — same model as
  our other CLI tools (`claude`, `codex`, etc., are also system-level).
  Discovered via `shutil.which("docker")` + `docker info` ping.
- **Base image**: ~500MB image with all agent CLIs preinstalled. Built
  on first use, cached. Image refresh needed when CLI versions change.
- **Auth credentials**: each CLI's auth (`~/.config/claude`, `~/.codex`,
  etc.) needs to be mounted read-only into the container or passed via
  env vars.
- **Tool server bridge**: the HTTP tool server runs on the host. The
  container needs `--add-host host.docker.internal:host-gateway` and
  the tool wrappers need to dial that hostname instead of `localhost`.
  ~20 lines.
- **Startup overhead**: 1-2s per `docker run`. Fine for one-shot
  per-request flow; meaningful overhead for chatty multi-turn loops.

### Why we haven't built it yet

The current sandbox quality is good enough for trusted local use on a
dev machine. The cases where containerization actually matters are:

- Running attempts in CI with untrusted task code.
- Parallel attempts on a shared machine with sensitive sibling work.
- Public-facing automation with adversarial prompts.

For these, build the wrapper backend per the sketch above. For solo
dev work on your own machine, the current setup is fine.

## Testing the sandbox

The probe utility in `tools/probe_agents.py` runs each backend through a
nine-operation gauntlet and reports the agent's self-report alongside
the filesystem ground truth. Run it before pushing changes that touch
sandbox machinery:

```
uv run tools/probe_agents.py            # all available backends
uv run tools/probe_agents.py codex_cli  # one backend
```

Output lands in `probe_results/<timestamp>/` with per-backend verdicts
and a SUMMARY.md.
