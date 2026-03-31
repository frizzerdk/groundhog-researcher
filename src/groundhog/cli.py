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
        # Set up uv project with dependencies
        subprocess.run(["uv", "init", "--no-readme"], cwd=target, capture_output=True)
        # Remove default main.py created by uv init
        (target / "main.py").unlink(missing_ok=True)
        # Add dependencies
        deps = ["groundhog-researcher", "python-dotenv"] + template.get("deps", [])
        subprocess.run(["uv", "add"] + deps, cwd=target, capture_output=True)

    # Copy template files (after uv init so task.py overwrites the default)
    for dest_name, src_name in template["files"].items():
        shutil.copy2(TEMPLATES_DIR / src_name, target / dest_name)

    if template.get("env"):
        (target / ".env").write_text("GEMINI_API_KEY=your-key-here\n")

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


def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help", "help"):
        print("Usage:")
        for name, info in TEMPLATES.items():
            print(f"  groundhog {name:12s} [directory]   {info['description']}")
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
    else:
        print(f"Unknown command: {cmd}")
        print("Try: groundhog --help")
        sys.exit(1)
