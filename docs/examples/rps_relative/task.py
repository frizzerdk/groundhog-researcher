# /// script
# dependencies = ["groundhog-researcher", "python-dotenv"]
# ///
"""Relative evaluation, end to end — a rock-paper-scissors tournament.

There is no absolute score for an RPS policy: "good" only means "beats the
others". So the Evaluator scores each candidate by playing it against a
reference pool and reporting its **win-rate**. The pool is the fixed baseline
strategy files plus every prior candidate in `toolkit.history` — which is why
this file exists as an example: it shows the one wire that turns a normal
Evaluator into a relative one.

    task.evaluator.pool = tk.history        # in build_toolkit(), below

See docs/relative_evaluation.md for the pattern, the helpers, and the pitfalls.

A candidate is a solution.py exposing:

    move(my_moves, opp_moves) -> "R" | "P" | "S"

where the two lists are the moves played so far this match (oldest first).

Run the offline demo (no LLM needed — scores the baselines against each other):

    uv run task.py
"""

from pathlib import Path

from groundhog import (
    Task, Data, Context, Evaluator, EvalStage, StageResult,
    Toolkit, assemble_toolkit,
)
from groundhog.utils.relative import round_robin

HERE = Path(__file__).parent
BASELINES = HERE / "baselines"

_BEATS = {"R": "P", "P": "S", "S": "R"}  # value beats key


# --- The problem -------------------------------------------------------------

class RPSData(Data):
    """A match is `rounds` throws; `games` matches per pairing (averaged, with
    seats swapped) to damp seed/seat variance. Small enough to run instantly."""

    def get_train(self):
        return {"rounds": 50, "games": 2}

    def get_test(self):
        return self.get_train()


class RPSContext(Context):
    def get_brief(self):
        return ('Write move(my_moves, opp_moves) -> "R" | "P" | "S". '
                "Beat the pool of opponent strategies.")

    def get_extended(self):
        return (
            "my_moves / opp_moves are the throws played so far this match "
            "(oldest first), each a letter in R/P/S. R beats S, S beats P, "
            "P beats R. You are scored by win-rate over a reference pool of "
            "opponents (fixed baselines + every prior candidate)."
        )

    def get_scoring(self):
        return ("Score = win-rate across all games vs. the reference pool "
                "(1 per round won, 0.5 per tie). There is no absolute target; "
                "beat the field.")


def _read_code(code_or_path):
    """The candidate's source, from a code string or a (workspace) path."""
    if isinstance(code_or_path, (str, bytes)) and "\n" in str(code_or_path):
        return str(code_or_path)
    p = Path(code_or_path)
    return (p / "solution.py").read_text(encoding="utf-8") if p.is_dir() \
        else p.read_text(encoding="utf-8")


def _load_move(code_or_path):
    """Return the move() callable from a code string or a workspace path."""
    ns = {}
    exec(_read_code(code_or_path), ns)
    return ns["move"]


def _round_score(a, b):
    """a's score for one throw pair: 1 win, 0.5 tie, 0 loss.
    a wins when a is the move that beats b (_BEATS[b] == a)."""
    if a == b:
        return 0.5
    return 1.0 if _BEATS[b] == a else 0.0


def _play_match(move_a, move_b, rounds):
    """Play one match; return move_a's mean per-round score in [0, 1]."""
    hist_a, hist_b = [], []
    total = 0.0
    for _ in range(rounds):
        # Each policy sees its own moves first, the opponent's second.
        a = _clean(move_a(list(hist_a), list(hist_b)))
        b = _clean(move_b(list(hist_b), list(hist_a)))
        total += _round_score(a, b)
        hist_a.append(a)
        hist_b.append(b)
    return total / rounds


def _clean(m):
    return m if m in ("R", "P", "S") else "R"  # a broken throw forfeits the round


class TournamentEvaluator(Evaluator):
    """Relative Evaluator: score = win-rate over the reference pool.

    `pool` is injected by build_toolkit() (`= tk.history`). The reference pool
    is the fixed baseline files plus every committed prior candidate. The
    candidate never plays itself: during optimization it isn't in history yet,
    and when a committed attempt (or a baseline file) is re-evaluated, any
    rival with byte-identical code is excluded from its pool.
    """

    pool = None  # set to toolkit.history in build_toolkit(); None until wired

    def _reference(self, exclude_code=None):
        """(label, move_fn) opponents: fixed baselines + committed candidates,
        minus any rival whose code equals the candidate's (self-exclusion)."""
        opps = []
        for p in sorted(BASELINES.glob("*.py")):
            code = p.read_text(encoding="utf-8")
            if code == exclude_code:
                continue
            opps.append((f"baseline:{p.stem}", _load_move(code)))
        if self.pool is not None:
            for a in self.pool.list():                 # committed attempts only
                code = a.read_file("solution.py")
                if code and code != exclude_code:
                    try:
                        opps.append((a.id, _load_move(code)))
                    except Exception:                  # a broken rival is skipped
                        continue
        return opps

    def evaluate(self, code_or_path, data):
        try:
            cand_code = _read_code(code_or_path)
            cand = _load_move(cand_code)
        except Exception as e:  # noqa: BLE001 — task contract: errors -> StageResult
            return StageResult(errors={"load": str(e)})

        d = data.get_test()
        rounds, games = d["rounds"], d["games"]

        def play(candidate_fn, opponent):
            _label, opp_fn = opponent
            # Symmetric seats: candidate first, then swapped, averaged.
            a = _play_match(candidate_fn, opp_fn, rounds)
            b = 1.0 - _play_match(opp_fn, candidate_fn, rounds)
            return (a + b) / 2.0

        r = round_robin(cand, self._reference(exclude_code=cand_code),
                        play_fn=play, games_per_pair=games,
                        key=lambda o: o[0])
        return StageResult(metrics={
            "win_rate": r["win_rate"],
            "games": r["games"],
            "per_opponent": r["per_opponent"],
        })

    def get_stages(self, data):
        return [
            EvalStage("tournament", "win-rate vs. reference pool",
                      lambda cp, d=data: self.evaluate(cp, d),
                      scorer=lambda res: -1.0 if res.errors
                      else res.metrics["win_rate"])
        ]


task = Task(data=RPSData(), context=RPSContext(),
            evaluator=TournamentEvaluator(), name="RPS-Tournament")


# --- The bench: the one wire that makes the Evaluator relative ---------------

def build_toolkit() -> Toolkit:
    tk = assemble_toolkit(task, path=HERE, through="tournament")
    task.evaluator.pool = tk.history      # <-- Evaluator now sees prior candidates
    return tk


# --- Offline demo: no LLM, no history — just rank the baselines --------------

if __name__ == "__main__":
    # Score each baseline *as if* it were a candidate, against the whole pool
    # (its rivals are the OTHER baselines — self is excluded). No optimizer,
    # no API calls — this just demonstrates the Evaluator and the
    # non-transitivity of RPS.
    for f in sorted(BASELINES.glob("*.py")):
        res = task.evaluate(f)
        m = res.stages["tournament"].metrics
        per = " ".join(f"{k.split(':')[-1]}={v:.2f}"
                       for k, v in m["per_opponent"].items())
        print(f"{f.stem:15s} win_rate={m['win_rate']:.3f}  [{per}]")
    print("\nthe constant baselines form a non-transitive cycle (paper beats "
          "rock beats scissors beats paper): pairwise wins cannot rank them. "
          "A win-rate over the whole pool, not any one head-to-head, is the "
          "stable signal.")
