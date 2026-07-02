# Examples

Two reference run dirs, deliberately at opposite ends:

| | backend | learnings | LLM | maintenance |
|---|---|---|---|---|
| **[01_simplest](01_simplest/)** | folder | `learnings.md` | none (deterministic strategy) | frozen — touched almost never |
| **[02_recommended](02_recommended/)** | git store | Compacted (condensed view + queue) | auto-discovered | living — updated with every new feature |

Both follow the run-dir contract: `task.py` defines the task and exposes
`def build_toolkit() -> Toolkit` (assemble + configure, never run), so the
CLI and agents can load them without side effects:

```
cd examples/01_simplest
python task.py 5              # run 5 iterations
groundhog attempt list        # inspect without running anything
groundhog tool list           # every tool on this run's toolkit
```
