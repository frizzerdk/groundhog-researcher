"""Baseline: cycle R -> P -> S -> R ... regardless of the opponent."""


def move(my_moves, opp_moves):
    return ["R", "P", "S"][len(my_moves) % 3]
