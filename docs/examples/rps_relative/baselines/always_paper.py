"""Baseline: always throw Paper. With always_rock and always_scissors it
forms the textbook non-transitive cycle: paper beats rock, rock beats
scissors, scissors beats paper."""


def move(my_moves, opp_moves):
    return "P"
