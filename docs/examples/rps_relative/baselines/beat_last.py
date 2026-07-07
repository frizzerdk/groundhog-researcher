"""Baseline: throw the move that would have beaten the opponent's last throw.
Exploits any policy that repeats itself; loses to anything that anticipates it."""

_BEATS = {"R": "P", "P": "S", "S": "R"}  # value beats the key


def move(my_moves, opp_moves):
    if not opp_moves:
        return "R"
    return _BEATS[opp_moves[-1]]
