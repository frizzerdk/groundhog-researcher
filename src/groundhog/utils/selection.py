"""Prior selection — potential-weighted random across trunk leaders.

Vault: Strategy — Selection.md, Selection_PotentialScore.md

Selects which attempt to build from next. High-scoring trunks get picked more,
but short/unexplored trunks get an exploration bonus. Long, well-explored
trunks get deprioritized.
"""

import math
import random as rand_module
from typing import Callable, List, Optional

from groundhog.base.attempt_history import Attempt, AttemptHistory
from groundhog.base.types import StageResult


def select_prior(history: AttemptHistory, scorer: Callable[[StageResult], float],
                 rng: rand_module.Random = None) -> Optional[Attempt]:
    """Select a prior using potential-weighted random across trunk leaders.

    Returns the best attempt from a probabilistically chosen trunk.
    High-scoring trunks are favored, but unexplored trunks get a bonus.
    """
    trunks = history.derive_trunks(scorer)
    if not trunks:
        return None

    rng = rng or rand_module.Random()
    leaders = [(trunk[-1], len(trunk)) for trunk in trunks]
    # Filter out failed trunks
    leaders = [(a, l) for a, l in leaders if _score(a, scorer) > 0]
    if not leaders:
        return None

    global_best = max(_score(a, scorer) for a, _ in leaders)

    potentials = []
    for attempt, trunk_len in leaders:
        score = _score(attempt, scorer)
        score_component = max(score / global_best, 0.01) if global_best > 0 else 0.01
        unexplored_component = math.exp(-0.1 * (trunk_len - 1))
        potential = score_component + 0.3 * unexplored_component
        potentials.append(max(potential ** 2, 0.001))

    selected = rng.choices([a for a, _ in leaders], weights=potentials, k=1)[0]
    return selected


def get_trunk_leaders(history: AttemptHistory, scorer: Callable[[StageResult], float],
                      exclude: Optional[int] = None) -> List[Attempt]:
    """Get the best attempt from each trunk, optionally excluding one trunk."""
    trunks = history.derive_trunks(scorer)
    leaders = []
    for trunk in trunks:
        leader = trunk[-1]
        if exclude is not None and leader.number == exclude:
            continue
        leaders.append(leader)
    return leaders


def _score(attempt: Attempt, scorer) -> float:
    result = attempt.result
    if not result.completed:
        return -1.0
    last = list(result.stages.values())[-1]
    return scorer(last)
