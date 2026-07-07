"""Groundhog CLI — scaffold task folders from templates.

Usage:
    groundhog init [directory]              # basic template, full uv project
    groundhog init-llm [directory]          # detailed template for LLM agents
    groundhog init-mock [directory]         # mock task — no LLM, for testing
    groundhog init-mnist [directory]        # MNIST example — real ML task

    groundhog init --script [directory]     # script-only (no project, inline deps)
"""

import shutil
import sys
from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent / "templates"
SKILLS_DIR = Path(__file__).parent / "skills"

TEMPLATES = {
    "init": {
        "description": "Basic task template with brief comments",
        "files": {"task.py": "basic.py"},
        "env": True,
    },
    "init-llm": {
        "description": "Detailed template with full guide for LLM agents",
        "files": {"task.py": "llm_guide.py"},
        "env": True,
    },
    "init-mock": {
        "description": "Mock task — deterministic, no LLM needed, for testing",
        "files": {"task.py": "mock_task.py", "mock_strategy.py": "mock_strategy.py"},
        "env": False,
    },
    "init-mnist": {
        "description": "MNIST digit classification — real ML task with 50 training samples",
        "files": {"task.py": "mnist_task.py"},
        "deps": ["numpy", "scikit-learn", "torch"],
        "env": True,
    },
}


def init(template_name, target_dir=None, script_only=False):
    template = TEMPLATES[template_name]
    target = Path(target_dir) if target_dir else Path("my_task")

    # A lone .claude/ (e.g. from an earlier `groundhog skills install`)
    # doesn't count as content worth protecting from scaffolding.
    if target.exists() and any(p.name != ".claude" for p in target.iterdir()):
        print(f"Directory '{target}' already exists and is not empty.")
        return 1

    target.mkdir(parents=True, exist_ok=True)

    if not script_only:
        # Write pyproject.toml directly (no subprocess needed)
        deps = ["groundhog-researcher", "python-dotenv"] + template.get("deps", [])
        deps_str = ", ".join(f'"{d}"' for d in deps)
        name = target.name.replace(" ", "-").lower()
        # Check if groundhog-researcher is on real PyPI
        _on_pypi = False
        try:
            import urllib.request
            urllib.request.urlopen("https://pypi.org/pypi/groundhog-researcher/json", timeout=3)
            _on_pypi = True
        except Exception:
            pass

        pyproject_content = (
            f'[project]\nname = "{name}"\nversion = "0.1.0"\n'
            f'requires-python = ">=3.11"\n'
            f'dependencies = [{deps_str}]\n'
        )
        if not _on_pypi:
            pyproject_content += (
                '\n[[tool.uv.index]]\nname = "testpypi"\n'
                'url = "https://test.pypi.org/simple/"\n'
                'explicit = true\n'
                '\n[tool.uv.sources]\n'
                'groundhog-researcher = { index = "testpypi" }\n'
            )
        (target / "pyproject.toml").write_text(pyproject_content, encoding="utf-8")

    # Copy template files (after uv init so task.py overwrites the default)
    for dest_name, src_name in template["files"].items():
        shutil.copy2(TEMPLATES_DIR / src_name, target / dest_name)

    if template.get("env"):
        (target / ".env").write_text("# Add API keys here (optional - auto_registry finds CLI tools automatically)\n# ANTHROPIC_API_KEY=\n# OPENAI_API_KEY=\n# GEMINI_API_KEY=\n", encoding="utf-8")

    # Skills travel with the package: every run dir gets the interactive-
    # session skills so `/groundhog-fresh` etc. work out of the box.
    install_skills(target, quiet=True)

    mode = "script" if script_only else "project"
    print(f"Created {mode} in {target}/")
    print(f"  {template['description']}")
    print()
    print("Next steps:")
    print(f"  cd {target}")
    if template.get("env"):
        print("  # edit .env with your API key")
    print("  # edit task.py with your task logic")
    print("  uv run task.py 10")
    return 0


def install_skills(target_dir=None, quiet=False) -> int:
    """Copy the packaged groundhog skills into ``<target>/.claude/skills/``.

    Idempotent and safe to re-run after upgrading groundhog — each
    groundhog skill dir is overwritten with the packaged version. Only
    the packaged skill names are touched: non-groundhog skills already
    in the folder are never modified or removed.
    """
    target = Path(target_dir) if target_dir else Path.cwd()
    if not SKILLS_DIR.exists():
        print(f"No packaged skills found at {SKILLS_DIR} (broken install?)")
        return 1
    names = sorted(p.name for p in SKILLS_DIR.iterdir() if p.is_dir())
    if not names:
        print(f"No packaged skills found at {SKILLS_DIR} (broken install?)")
        return 1
    dest_root = target / ".claude" / "skills"
    dest_root.mkdir(parents=True, exist_ok=True)
    for name in names:
        # Replace, don't merge: a file the packaged skill dropped must not
        # survive a re-install as a stale leftover.
        dest = dest_root / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(SKILLS_DIR / name, dest)
    if not quiet:
        print(f"Installed {len(names)} groundhog skills into {dest_root}")
        for name in names:
            print(f"  {name}")
        print("Invoke them in a Claude Code session opened in the run dir, "
              "e.g. /groundhog-fresh")
    return 0


def skills_group(args) -> int:
    """`groundhog skills <subcommand>` — currently just `install`."""
    if not args or args[0] in ("-h", "--help"):
        print("Usage:")
        print("  groundhog skills install [directory]   "
              "Copy the packaged skills into <dir>/.claude/skills/")
        print()
        print("Idempotent: re-run after upgrading groundhog to refresh them. "
              "Every `groundhog init*` also installs them.")
        return 0
    if args[0] == "install":
        return install_skills(args[1] if len(args) > 1 else None)
    print(f"Unknown skills subcommand: {args[0]!r}. Try: groundhog skills --help")
    return 1


def _backend_source(name, backend):
    """Describe how this backend connects — CLI, API key, or local server."""
    cli_backends = {"claude_code": "claude CLI", "copilot": "copilot CLI",
                    "gemini_cli": "gemini CLI", "opencode": "opencode CLI"}
    if name in cli_backends:
        return cli_backends[name]
    if name == "ollama":
        return "localhost:11434"
    # API key backends
    key_map = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY",
               "gemini": "GEMINI_API_KEY", "openrouter": "OPENROUTER_API_KEY",
               "deepseek": "DEEPSEEK_API_KEY", "groq": "GROQ_API_KEY"}
    if name in key_map:
        return key_map[name]
    return backend.__class__.__name__


def _backend_source_from_class(backend):
    """Infer source from backend class and attributes."""
    cls = backend.__class__.__name__
    if cls == "ClaudeCodeBackend":
        return "claude CLI"
    if cls == "CopilotBackend":
        return "copilot CLI"
    if cls == "GeminiCLIBackend":
        return "gemini CLI"
    if cls == "OpenCodeBackend":
        return "opencode CLI"
    if cls == "AnthropicBackend":
        return "ANTHROPIC_API_KEY"
    if cls == "GeminiBackend":
        return "GEMINI_API_KEY"
    if cls == "OpenAICompatibleBackend":
        url = getattr(backend, 'base_url', '')
        if 'openai.com' in url:
            return "OPENAI_API_KEY"
        if 'openrouter' in url:
            return "OPENROUTER_API_KEY"
        if 'deepseek' in url:
            return "DEEPSEEK_API_KEY"
        if 'groq.com' in url:
            return "GROQ_API_KEY"
        if 'localhost' in url:
            return url.replace('/v1', '')
        return url
    return cls


COMPONENTS = {
    "strategy": {"template": "strategy.py", "default_name": "strategy.py"},
    "backend":  {"template": "backend.py",  "default_name": "backend.py"},
}


def new_component(args):
    """Generate a component template file."""
    if not args or args[0] in ("-h", "--help"):
        print("Usage: groundhog new <component> [filename]")
        print()
        print("Components:")
        print("  strategy    Custom strategy with Config, composed method, retries")
        print("  backend     Custom LLM backend (API or CLI subprocess)")
        return 0

    component = args[0]
    if component not in COMPONENTS:
        print(f"Unknown component: {component}")
        print(f"Available: {', '.join(COMPONENTS.keys())}")
        return 1

    info = COMPONENTS[component]
    target = Path(args[1]) if len(args) > 1 else Path(info["default_name"])

    if target.exists():
        print(f"File '{target}' already exists.")
        return 1

    shutil.copy2(TEMPLATES_DIR / info["template"], target)

    print(f"Created {component} template: {target}")
    print("  Edit the file and customize the logic.")
    return 0


def show_backends():
    """Show available LLM backends and auto_registry tier assignments."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from groundhog.backends.discover import discover_backends, auto_registry, _auth_warnings, _load_preferences

    prefs = _load_preferences()
    if prefs:
        print("Preferences (~/.groundhog/config.json):")
        if "prefer" in prefs:
            print(f"  prefer: {prefs['prefer']}")
        for tier, override in prefs.get("tiers", {}).items():
            model = override.get("model", "")
            print(f"  tier {tier}: {override['backend']} {model}".rstrip())
        print()

    found = discover_backends()

    if _auth_warnings:
        for name, msg in _auth_warnings:
            if name == "copilot":
                print("[copilot] Found but not authenticated.")
                answer = input("  Log in now? (y/n): ").strip().lower()
                if answer in ("y", "yes", ""):
                    from groundhog.backends.copilot import login_copilot
                    if login_copilot():
                        print("  Logged in successfully.")
                        # Re-run discovery with new auth
                        found = discover_backends()
                    else:
                        print("  Login failed.")
                print()
            else:
                print(f"[{name}] {msg}")
                print()

    if not found:
        print("No LLM backends found.")
        print()
        print("To get started, do one of:")
        print("  - Install Claude Code: https://claude.ai/code")
        print("  - Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY in .env")
        print("  - Start Ollama: https://ollama.com")
        return 0

    print("Available backends:")
    for name, backend in found.items():
        source = _backend_source(name, backend)
        print(f"  {name:15s} {backend.model:40s} ({source})")

    print()
    reg = auto_registry()
    if reg is None:
        print("Auto-registry: no backends discoverable")
    else:
        print("Auto-registry tier assignments:")
        for tier in ["max", "high", "default", "budget", "cheap"]:
            b = reg.get(tier)
            source = _backend_source_from_class(b)
            print(f"  {tier:10s} {b.model:40s} ({source})")

    return 0


def set_prefer(args):
    """Set or reset global backend preference."""
    from groundhog.backends.discover import _load_preferences, _save_preferences

    if not args:
        print("Usage: groundhog prefer <backend>")
        print("       groundhog prefer reset")
        return 1

    if args[0] == "reset":
        _save_preferences({})
        print("Preferences reset.")
        return 0

    backend_name = args[0]

    # If preferring copilot, check auth and offer login
    if backend_name == "copilot" and shutil.which("copilot"):
        from groundhog.backends.copilot import check_copilot_auth, login_copilot
        ok, _ = check_copilot_auth()
        if not ok:
            print("Copilot is not authenticated.")
            answer = input("Log in now? (y/n): ").strip().lower()
            if answer in ("y", "yes", ""):
                if login_copilot():
                    print("Logged in successfully.")
                else:
                    print("Login failed.")
                    return 1
            else:
                print("Skipped. Set COPILOT_GITHUB_TOKEN in .env for auth.")

    prefs = _load_preferences()
    prefs["prefer"] = backend_name
    _save_preferences(prefs)
    print(f"Preferred backend: {backend_name}")
    return 0


def set_prefer_tier(args):
    """Set or reset a specific tier override."""
    from groundhog.backends.discover import _load_preferences, _save_preferences

    if not args or len(args) < 2:
        print("Usage: groundhog prefer-tier <tier> <backend> [model]")
        print("       groundhog prefer-tier reset [tier]")
        print("Tiers: max, high, default, budget, cheap")
        return 1

    if args[0] == "reset":
        prefs = _load_preferences()
        if len(args) > 1:
            prefs.get("tiers", {}).pop(args[1], None)
            if not prefs.get("tiers"):
                prefs.pop("tiers", None)
        else:
            prefs.pop("tiers", None)
        _save_preferences(prefs)
        print("Tier preferences reset.")
        return 0

    tier = args[0]
    backend_name = args[1]
    model = args[2] if len(args) > 2 else None

    prefs = _load_preferences()
    tiers = prefs.setdefault("tiers", {})
    entry = {"backend": backend_name}
    if model:
        entry["model"] = model
    tiers[tier] = entry
    _save_preferences(prefs)

    label = f"{backend_name} {model}" if model else backend_name
    print(f"Tier {tier}: {label}")
    return 0


# ---------------------------------------------------------------------------
# attempt / eval commands — manual attempt lifecycle + scoring (LLM-free).
#
# These call the rundir loader to get the run's task + attempt history, then
# drive the AttemptHistory / Task.evaluate APIs directly. No optimizer is run.
# ---------------------------------------------------------------------------


def _resolve_run(args=None):
    """Load the run in the cwd. Returns a LoadedRun or None (after printing an
    error). ``args`` is accepted for forward-compat (-C/--run-dir) but v1 only
    uses the cwd."""
    from groundhog import rundir

    run_dir = None
    try:
        return rundir.load_run(run_dir=run_dir)
    except FileNotFoundError as e:
        print(str(e))
        return None
    except SystemExit as e:
        # An ill-behaved task.py (module-level sys.exit) must not kill the CLI.
        print(f"Could not load this run: task.py exited at import ({e})")
        return None
    except Exception as e:  # noqa: BLE001 — surface a clean error to the CLI user
        print(f"Could not load this run: {e}")
        return None


def _scorer_for(task, through=None):
    """Replicate SimpleOptimizer._get_scorer: the final stage's scorer."""
    stages = task.evaluator.eval_stages(task.data, through=through)
    return stages[-1].score


def _score_result(result, scorer):
    """Score an EvaluationResult read-side: scorer of the last stage, or -1.0
    when the result did not complete (a failed gate)."""
    if not result.completed or not result.stages:
        return -1.0
    last = list(result.stages.values())[-1]
    return scorer(last)


def _short(value, n=8):
    """Short id form (git shas are long; folder ids are already short)."""
    if value is None:
        return "-"
    s = str(value)
    return s[:n] if len(s) > n else s


def _print_stage_scores(result, scorer):
    """Print per-stage score + the final overall line."""
    for name, stage in result.stages.items():
        print(f"  {name}: score={scorer(stage):.4f}")
    overall = _score_result(result, scorer)
    if result.completed:
        print(f"  overall: {overall:.4f}  (completed)")
    else:
        print(f"  overall: {overall:.4f}  (FAILED at stage {result.failed_stage})")


def _attempt_score(attempt, scorer):
    try:
        return _score_result(attempt.result, scorer)
    except Exception:
        return -1.0


def attempt_group(args):
    """`groundhog attempt <subcommand>` — manual attempt lifecycle."""
    usage = (
        "Usage: groundhog attempt <subcommand>\n"
        "\n"
        "  new [--fresh] [--parent ID] [--no-seed] [--name NAME]\n"
        "                                Open a workspace (--fresh: parentless,\n"
        "                                founds a new family; default parent = best)\n"
        "  list [--all]                                  List attempts (--all incl. failed)\n"
        "  show <id> [--file F]                          Show an attempt (or one file)\n"
        "  in-progress                                   List open workspaces\n"
        "  resume <wsid>                                 Re-acquire an open workspace\n"
        "  commit <wsid> [--fail] [--eval] [--through S] Finalize a workspace\n"
        "  abort <wsid>                                  Discard an open workspace\n"
        "  reap [--ttl S]                                Abort crashed workspaces\n"
        "  best                                          Show the best attempt\n"
    )
    if not args or args[0] in ("-h", "--help"):
        print(usage)
        return 0

    sub = args[0]
    rest = args[1:]

    handlers = {
        "new": _attempt_new,
        "list": _attempt_list,
        "show": _attempt_show,
        "in-progress": _attempt_in_progress,
        "resume": _attempt_resume,
        "commit": _attempt_commit,
        "abort": _attempt_abort,
        "reap": _attempt_reap,
        "best": _attempt_best,
    }
    handler = handlers.get(sub)
    if handler is None:
        print(f"Unknown attempt subcommand: {sub}")
        print(usage)
        return 1
    return handler(rest)


def _flag(args, name):
    """Pop a boolean flag; return (present, remaining_args)."""
    present = name in args
    return present, [a for a in args if a != name]


def _opt(args, name):
    """Pop an option with a value (``--name VALUE``); return (value, remaining)."""
    if name in args:
        i = args.index(name)
        if i + 1 < len(args):
            value = args[i + 1]
            return value, args[:i] + args[i + 2:]
    return None, args


def _attempt_new(args):
    no_seed, args = _flag(args, "--no-seed")
    fresh, args = _flag(args, "--fresh")
    parent, args = _opt(args, "--parent")
    name, args = _opt(args, "--name")

    run = _resolve_run()
    if run is None:
        return 1
    history, task = run.history, run.task

    # --fresh opens a PARENTLESS workspace — how a new family is founded,
    # matching the automated fresh strategy (prior=None). This is what makes
    # the commit-time gates classify it as fresh: the gated finish decides
    # fresh-vs-child by the parent pointer, and a defaulted best-parent
    # would silently restore the parent's direction over the new one.
    # (--parent none is the legacy spelling of the same thing.)
    if parent == "none":
        fresh = True
        parent = None

    try:
        # Default parent = current best, if any attempts exist.
        if parent is None and not fresh:
            scorer = _scorer_for(task, through=getattr(run.toolkit, "through", None))
            best = history.best(scorer)
            parent = best.id if best else None

        ws = history.workspace(parent=parent)

        if parent is not None and not no_seed:
            parent_attempt = history.get(parent)
            if parent_attempt is None:
                ws.abort()
                print(f"No such parent attempt: {parent}")
                return 1
            history.seed_from_parent(ws, parent_attempt)

        if name:
            ws.name = name

        print(f"Opened workspace {ws.display_id}")
        print(f"  path:   {ws.path}")
        if parent is not None:
            print(f"  parent: {parent}")
        if fresh:
            print("  fresh:  no parent — write core_direction.md (first line = "
                  "approach name) before committing")
        print(f"  edit solution.py, then: groundhog attempt commit {ws.display_id} --eval")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"Could not open a workspace: {e}")
        return 1


def _attempt_list(args):
    show_all, args = _flag(args, "--all")
    run = _resolve_run()
    if run is None:
        return 1
    history, task = run.history, run.task
    scorer = _scorer_for(task, through=getattr(run.toolkit, "through", None))

    attempts = history.list(only_done=not show_all)
    if not attempts:
        print("No attempts yet.")
        return 0

    print(f"{'id':<10} {'parent':<10} {'status':<12} {'score':<9} name")
    for a in attempts:
        if a.status == "done":
            score = _attempt_score(a, scorer)
            score_str = f"{score:.4f}"
        else:
            score_str = "-"
        print(f"{_short(a.id):<10} {_short(a.parent):<10} {a.status:<12} "
              f"{score_str:<9} {a.name}")
    return 0


def _attempt_show(args):
    file_arg, args = _opt(args, "--file")
    if not args:
        print("Usage: groundhog attempt show <id> [--file F]")
        return 1
    attempt_id = args[0]

    run = _resolve_run()
    if run is None:
        return 1
    history, task = run.history, run.task

    attempt = history.get(attempt_id)
    if attempt is None:
        print(f"No such attempt: {attempt_id}")
        return 1

    if file_arg is not None:
        content = attempt.read_file(file_arg)
        if content is None:
            print(f"No such file in attempt {attempt_id}: {file_arg}")
            return 1
        print(content)
        return 0

    scorer = _scorer_for(task, through=getattr(run.toolkit, "through", None))
    print(f"id:      {attempt.id}")
    print(f"parent:  {attempt.parent}")
    print(f"status:  {attempt.status}")
    print(f"name:    {attempt.name}")
    print(f"created: {attempt.created_at}")
    cached = history.get_note(attempt.id, "score")
    if cached is not None:
        print(f"note[score]: {cached}   (mutable cache — canonical score is read-side)")
    print(f"metadata: {attempt.metadata}")
    print()
    print("stages:")
    try:
        result = attempt.result
        for name, stage in result.stages.items():
            print(f"  {name}: score={scorer(stage):.4f} metrics={stage.metrics}")
        print(f"  overall: {_score_result(result, scorer):.4f} "
              f"({'completed' if result.completed else 'failed at ' + str(result.failed_stage)})")
    except Exception as e:  # noqa: BLE001
        print(f"  (no readable result: {e})")
    print()
    print("files:")
    for f in attempt.list_files():
        print(f"  {f}")
    return 0


def _attempt_in_progress(args):
    import time

    run = _resolve_run()
    if run is None:
        return 1
    items = run.history.list_in_progress()
    if not items:
        print("No in-progress workspaces.")
        return 0

    now = time.time()
    print(f"{'wsid':<10} {'parent':<10} {'age(s)':<8} {'state':<9} path")
    for ip in items:
        age = int(now - ip.started_at)
        state = "live" if ip.live else "CRASHED"
        print(f"{_short(ip.workspace_id):<10} {_short(ip.parent):<10} "
              f"{age:<8} {state:<9} {ip.path}")
    return 0


def _attempt_resume(args):
    if not args:
        print("Usage: groundhog attempt resume <wsid>")
        return 1
    wsid = args[0]
    run = _resolve_run()
    if run is None:
        return 1
    try:
        ws = run.history.resume(wsid)
    except (KeyError, NotImplementedError) as e:
        print(f"Could not resume {wsid}: {e}")
        return 1
    print(f"Resumed workspace {ws.display_id}")
    print(f"  path: {ws.path}")
    print(f"  edit, then: groundhog attempt commit {ws.display_id}")
    return 0


def _attempt_commit(args):
    do_fail, args = _flag(args, "--fail")
    do_eval, args = _flag(args, "--eval")
    through, args = _opt(args, "--through")
    strategy, args = _opt(args, "--strategy")
    strategy = strategy or "manual"
    # A dangling option (e.g. `--strategy` with no value) or an unknown flag
    # must not silently commit mislabeled work.
    if not args or any(a.startswith("--") for a in args) or len(args) > 1:
        print("Usage: groundhog attempt commit <wsid> [--fail] [--eval] "
              "[--through STAGE] [--strategy LABEL]")
        return 1
    wsid = args[0]

    run = _resolve_run()
    if run is None:
        return 1
    history, task, toolkit = run.history, run.task, run.toolkit

    try:
        ws = history.resume(wsid)
    except (KeyError, NotImplementedError) as e:
        print(f"Could not resume {wsid}: {e}")
        return 1

    through = through or getattr(toolkit, "through", None)
    scorer = _scorer_for(task, through=through)
    parent_id = getattr(ws, "parent", None)
    prior = history.get(parent_id) if parent_id else None

    try:
        if do_eval:
            # The standard finish: evaluate, then gates -> record -> commit
            # -> score note, exactly as the automated strategies finish.
            from groundhog.utils.finalize import finalize_attempt

            result = task.evaluate(ws.path, through=through)
            print("Evaluation:")
            _print_stage_scores(result, scorer)
            if do_fail:
                # The user's verdict: record real work as failed. Shape it
                # like every other failed record (a failed stage with the
                # reason), not a bare completed=false.
                from groundhog.utils.direction import mark_result_failed
                mark_result_failed(result, "manual",
                                   "committed as failed by --fail")
            attempt = finalize_attempt(
                toolkit, ws, result, prior, strategy=strategy
            )
            _print_gate_outcome(attempt.metadata)
        else:
            # No evaluation of record — but the gates and the producer
            # metadata apply to every commit. Composed from the same public
            # pieces the standard finish uses (result.json stays eval-only).
            from groundhog.utils.direction import (
                inherit_direction_from_attempt,
                promote_workspace_direction,
                workspace_name,
            )
            from groundhog.utils.gates import evaluate_gates
            from groundhog.utils.results import write_metadata

            if prior is None:
                promote_workspace_direction(ws.path)
            violations = evaluate_gates(
                ws.path, prior, history=history, exclude=[ws.display_id]
            )
            metadata = {
                "strategy": strategy,
                "prior": prior.id if prior else None,
                "cost": 0.0,
            }
            success = not do_fail
            for v in violations:
                if v.severity == "fail":
                    metadata["gate_failure"] = v.message
                    success = False
                elif v.gate == "direction-modified":
                    metadata["direction_restored"] = True
                elif v.gate == "solution-identical":
                    metadata["non_promotable"] = True
                    metadata["non_promotable_reason"] = v.message
            if prior is not None:
                inherit_direction_from_attempt(prior, ws.path)
            write_metadata(ws.path, metadata)
            if not ws.name:
                derived = workspace_name(ws.path)
                if derived:
                    ws.name = derived
            attempt = ws.commit(success=success)
            _print_gate_outcome(metadata)

        print(f"Committed attempt {attempt.id} ({attempt.status})")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"Commit failed: {e}")
        return 1


def _print_gate_outcome(metadata):
    """Surface what the gates decided so the session user is never surprised."""
    if metadata.get("gate_failure"):
        print(f"Gate: {metadata['gate_failure']} -> committed as failed")
    if metadata.get("non_promotable"):
        print(f"Gate: {metadata.get('non_promotable_reason', 'non-promotable')} "
              f"-> flagged non-promotable (commit stays done)")
    if metadata.get("direction_restored"):
        print("Gate: inherited core_direction.md was modified -> parent's "
              "direction restored")


def _attempt_abort(args):
    if not args:
        print("Usage: groundhog attempt abort <wsid>")
        return 1
    wsid = args[0]
    run = _resolve_run()
    if run is None:
        return 1
    try:
        ws = run.history.resume(wsid)
    except (KeyError, NotImplementedError) as e:
        print(f"Could not abort {wsid}: {e}")
        return 1
    ws.abort()
    print(f"aborted {wsid}")
    return 0


def _attempt_reap(args):
    ttl, args = _opt(args, "--ttl")
    run = _resolve_run()
    if run is None:
        return 1
    ttl_s = float(ttl) if ttl else 300.0
    n = run.history.reap_in_progress(ttl_s=ttl_s)
    print(f"reaped {n}")
    return 0


def _attempt_best(args):
    run = _resolve_run()
    if run is None:
        return 1
    history, task = run.history, run.task
    scorer = _scorer_for(task, through=getattr(run.toolkit, "through", None))
    best = history.best(scorer)
    if best is None:
        print("No attempts yet.")
        return 0
    print(f"id:    {best.id}")
    print(f"score: {_attempt_score(best, scorer):.4f}")
    print(f"name:  {best.name}")
    return 0


def cmd_eval(args):
    """`groundhog eval <path-or-attempt-id> [--through STAGE] [--json]`."""
    as_json, args = _flag(args, "--json")
    through, args = _opt(args, "--through")
    if not args or args[0] in ("-h", "--help"):
        print("Usage: groundhog eval <path-or-attempt-id> [--through STAGE] [--json]")
        return 0 if (args and args[0] in ("-h", "--help")) else 1
    target_arg = args[0]

    run = _resolve_run()
    if run is None:
        return 1
    history, task, toolkit = run.history, run.task, run.toolkit
    through = through or getattr(toolkit, "through", None)
    scorer = _scorer_for(task, through=through)

    # Resolve the target: an attempt id, a directory, or a .py file.
    target = None
    attempt = history.get(target_arg)
    if attempt is not None:
        target = attempt.code  # a code string
    else:
        p = Path(target_arg)
        if p.is_dir():
            target = p
        elif p.is_file() and p.suffix == ".py":
            target = p.read_text(encoding="utf-8")
        else:
            print(f"Cannot resolve eval target: {target_arg!r} "
                  f"(not an attempt id, directory, or .py file)")
            return 1

    try:
        result = task.evaluate(target, through=through)
    except Exception as e:  # noqa: BLE001
        print(f"Evaluation crashed: {e}")
        return 1

    if as_json:
        import json

        out = {
            "completed": result.completed,
            "failed_stage": result.failed_stage,
            "overall_score": _score_result(result, scorer),
            "stages": {
                name: {
                    "score": scorer(stage),
                    "metrics": stage.metrics,
                    "errors": stage.errors,
                    "warnings": stage.warnings,
                    "artifacts": list(stage.artifacts.keys()),
                }
                for name, stage in result.stages.items()
            },
        }
        print(json.dumps(out, indent=2, default=str))
    else:
        for name, stage in result.stages.items():
            print(f"[{name}] score={scorer(stage):.4f}")
            if stage.metrics:
                print(f"  metrics:   {stage.metrics}")
            if stage.errors:
                print(f"  errors:    {stage.errors}")
            if stage.warnings:
                print(f"  warnings:  {stage.warnings}")
            if stage.artifacts:
                print(f"  artifacts: {list(stage.artifacts.keys())}")
        overall = _score_result(result, scorer)
        if result.completed:
            print(f"overall: {overall:.4f}  (completed)")
        else:
            print(f"overall: {overall:.4f}  (FAILED at stage {result.failed_stage})")

    return 0 if result.completed else 2


def tool_group(args):
    """`groundhog tool list` / `groundhog tool run <name> [--attempt ID] [-p k=v ...]`.

    The CLI as a thin toolkit wrapper: every tool on ``toolkit.agent_tools``
    — framework default or the task.py hook's own — is runnable from the
    terminal. ``--attempt`` points ``toolkit.ws`` at a chosen attempt for the
    duration (same bracket the strategy uses), so workspace-relative tools
    read that attempt's files.
    """
    if not args or args[0] in ("-h", "--help"):
        print("Usage:")
        print("  groundhog tool list                                    List this run's tools")
        print("  groundhog tool run <name> [--attempt ID] [-p k=v ...]  Invoke one")
        return 0
    sub, rest = args[0], args[1:]

    run = _resolve_run()
    if run is None:
        return 1
    toolkit = run.toolkit
    tools = {t.name: t for t in getattr(toolkit, "agent_tools", [])}

    if sub == "list":
        if not tools:
            print("No tools on this toolkit (framework defaults are empty and "
                  "the task.py agent_tools hook returned none).")
            return 0
        width = max(len(n) for n in tools)
        for name, t in sorted(tools.items()):
            desc = (getattr(t, "description", "") or "").strip().splitlines()
            print(f"  {name:<{width}}  {desc[0] if desc else ''}")
        return 0

    if sub == "run":
        attempt_id, rest = _opt(rest, "--attempt")
        params = {}
        positional = []
        i = 0
        while i < len(rest):
            if rest[i] == "-p" and i + 1 < len(rest) and "=" in rest[i + 1]:
                k, v = rest[i + 1].split("=", 1)
                params[k] = v
                i += 2
            else:
                positional.append(rest[i])
                i += 1
        if not positional:
            print("Usage: groundhog tool run <name> [--attempt ID] [-p k=v ...]")
            return 1
        name = positional[0]
        tool = tools.get(name)
        if tool is None:
            print(f"No tool named {name!r}. Try: groundhog tool list")
            return 1

        from contextlib import nullcontext
        bracket = toolkit.ws.attempt(attempt_id) if attempt_id else nullcontext()
        try:
            with bracket:
                result = tool.execute(**params)
        except Exception as e:  # noqa: BLE001 — CLI surface, print and exit
            print(f"Could not run {name!r}: {e}")
            return 1
        if result.success:
            print(result.output)
            return 0
        print(f"Tool failed: {result.error}")
        return 2

    print(f"Unknown tool subcommand: {sub!r}. Try: groundhog tool --help")
    return 1


def main():
    args = sys.argv[1:]

    if args and args[0] in ("-v", "--version", "version"):
        from groundhog import __version__
        print(f"groundhog-researcher {__version__}")
        return

    if not args or args[0] in ("-h", "--help", "help"):
        print("Usage:")
        for name, info in TEMPLATES.items():
            print(f"  groundhog {name:12s} [directory]   {info['description']}")
        print()
        print("  groundhog new strategy [file]     Generate a custom strategy template")
        print("  groundhog new backend [file]      Generate a custom backend template")
        print("  groundhog backends                Show available LLM backends")
        print("  groundhog prefer <backend>        Prefer a backend for all tiers")
        print("  groundhog prefer-tier <tier> <backend> [model]")
        print("  groundhog prefer reset            Reset all preferences")
        print()
        print("  groundhog attempt <subcommand>    Manual attempt lifecycle (new/list/show/commit/...)")
        print("  groundhog eval <path-or-id>       Score a solution dir, .py file, or attempt")
        print("  groundhog tool list|run           Run any toolkit tool from the terminal")
        print("  groundhog bench run|compare       Offline strategy benchmark (deterministic, no API)")
        print("  groundhog skills install [dir]    Install the session skills into a run dir")
        print()
        print("Options:")
        print("  --script    Script-only mode (no uv project, uses inline deps)")
        print()
        print("Then:")
        print("  cd my_task")
        print("  uv run task.py 10                 Run 10 iterations")
        print("  uv run task.py status             Show current state")
        return

    # Parse --script flag
    script_only = "--script" in args
    args = [a for a in args if a != "--script"]

    cmd = args[0] if args else "init"
    if cmd in TEMPLATES:
        target = args[1] if len(args) > 1 else None
        sys.exit(init(cmd, target, script_only=script_only))
    elif cmd == "new":
        sys.exit(new_component(args[1:]))
    elif cmd == "backends":
        sys.exit(show_backends())
    elif cmd == "prefer":
        sys.exit(set_prefer(args[1:]))
    elif cmd == "prefer-tier":
        sys.exit(set_prefer_tier(args[1:]))
    elif cmd == "attempt":
        sys.exit(attempt_group(args[1:]))
    elif cmd == "eval":
        sys.exit(cmd_eval(args[1:]))
    elif cmd == "tool":
        sys.exit(tool_group(args[1:]))
    elif cmd == "bench":
        from groundhog.bench.cli import bench_group
        sys.exit(bench_group(args[1:]))
    elif cmd == "skills":
        sys.exit(skills_group(args[1:]))
    else:
        print(f"Unknown command: {cmd}")
        print("Try: groundhog --help")
        sys.exit(1)
