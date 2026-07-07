"""Baseline: always throw Rock. The simplest possible policy — and a useful
anchor, because anything that can't beat a constant is broken."""


def move(my_moves, opp_moves):
    return "R"
