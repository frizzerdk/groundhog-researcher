"""Groundhog CLI — scaffold task folders from templates.

Usage:
    groundhog init [directory]              # basic template, full uv project
    groundhog init-llm [directory]          # detailed template for LLM agents
    groundhog init-mock [directory]         # mock task — no LLM, for testing
    groundhog init-mnist [directory]        # MNIST example — real ML task

    groundhog init --script [directory]     # script-only (no project, inline deps)
"""

import shutil
import subprocess
import sys
from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent / "templates"

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

    if target.exists() and any(target.iterdir()):
        print(f"Directory '{target}' already exists and is not empty.")
        return 1

    target.mkdir(parents=True, exist_ok=True)

    if not script_only:
        # Write pyproject.toml directly (no subprocess needed)
        deps = ["groundhog-researcher", "python-dotenv"] + template.get("deps", [])
        deps_str = ", ".join(f'"{d}"' for d in deps)
        name = target.name.replace(" ", "-").lower()
        pyproject_content = (
            f'[project]\nname = "{name}"\nversion = "0.1.0"\n'
            f'requires-python = ">=3.11"\n'
            f'dependencies = [{deps_str}]\n'
            f'\n[[tool.uv.index]]\nname = "testpypi"\n'
            f'url = "https://test.pypi.org/simple/"\n'
            f'explicit = true\n'
            f'\n[tool.uv.sources]\n'
            f'groundhog-researcher = {{ index = "testpypi" }}\n'
        )
        (target / "pyproject.toml").write_text(pyproject_content, encoding="utf-8")

    # Copy template files (after uv init so task.py overwrites the default)
    for dest_name, src_name in template["files"].items():
        shutil.copy2(TEMPLATES_DIR / src_name, target / dest_name)

    if template.get("env"):
        (target / ".env").write_text("# Add API keys here (optional - auto_registry finds CLI tools automatically)\n# ANTHROPIC_API_KEY=\n# OPENAI_API_KEY=\n# GEMINI_API_KEY=\n", encoding="utf-8")

    mode = "script" if script_only else "project"
    print(f"Created {mode} in {target}/")
    print(f"  {template['description']}")
    print()
    print("Next steps:")
    print(f"  cd {target}")
    if template.get("env"):
        print("  # edit .env with your API key")
    print("  # edit task.py with your task logic")
    print(f"  uv run task.py 10")
    return 0


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
    print(f"  Edit the file and customize the logic.")
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
                print(f"[copilot] Found but not authenticated.")
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
    try:
        reg = auto_registry()
        print("Auto-registry tier assignments:")
        for tier in ["max", "high", "default", "budget", "cheap"]:
            b = reg.get(tier)
            source = _backend_source_from_class(b)
            print(f"  {tier:10s} {b.model:40s} ({source})")
    except RuntimeError as e:
        print(f"Auto-registry: {e}")

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
    else:
        print(f"Unknown command: {cmd}")
        print("Try: groundhog --help")
        sys.exit(1)
