"""Codegen: extract_code / SEARCH-REPLACE application.

Regression provenance: the trailing-whitespace fallback in _apply_diff used to
match against stripped code but replace in the UNSTRIPPED original, where the
needle doesn't occur — str.replace silently no-opped and the edit vanished
while extract_code still reported method="search_replace" (audit 2026-07-01,
bug #1). The fallback now splices the replacement into the original at the
matched line window.
"""

from groundhog.utils.codegen import extract_code, _apply_diff

import pytest


PRIOR = "def solve():\n    x = 1\n    return x\n"


def test_search_replace_happy_path():
    text = (
        "<<<<<<< SEARCH\n    x = 1\n=======\n    x = 2\n>>>>>>> REPLACE"
    )
    code, diff = extract_code(text, PRIOR)
    assert diff.method == "search_replace"
    assert "x = 2" in code
    assert "x = 1" not in code


def test_search_replace_trailing_whitespace_fallback_actually_applies():
    """The edit must LAND when prior code has trailing whitespace the search
    block lacks. Old behavior: silently returned the prior unchanged."""
    prior_ws = "def solve():\n    x = 1   \n    return x\n"  # trailing spaces on x=1
    text = (
        "<<<<<<< SEARCH\ndef solve():\n    x = 1\n=======\ndef solve():\n    x = 99\n>>>>>>> REPLACE"
    )
    code, diff = extract_code(text, prior_ws)
    assert diff.method == "search_replace"
    assert "x = 99" in code, "whitespace-fallback edit was silently dropped"
    assert "x = 1" not in code
    # Untouched lines keep their original bytes
    assert "    return x" in code


def test_search_replace_fallback_replaces_only_matched_window():
    # Multi-line search whose first line has trailing ws in the code but not
    # in the search → substring path misses, line-window fallback must fire
    # and leave every line outside the window byte-identical.
    prior_ws = "x = 0\na = 1  \nb = 2\nx = 0\n"
    applied = _apply_diff(prior_ws, [("a = 1\nb = 2", "a = 9\nb = 9")])
    assert applied == "x = 0\na = 9\nb = 9\nx = 0\n"


def test_search_not_found_falls_through_to_fenced():
    text = (
        "<<<<<<< SEARCH\nnot in prior\n=======\nreplacement\n>>>>>>> REPLACE\n"
        "```python\ndef solve():\n    return 42\n```"
    )
    code, diff = extract_code(text, PRIOR)
    assert diff.method == "fenced"
    assert "return 42" in code


def test_fenced_block_extraction():
    code, diff = extract_code("Here you go:\n```python\ndef f():\n    return 1\n```")
    assert diff.method == "fenced"
    assert code == "def f():\n    return 1"


def test_nothing_valid_returns_empty():
    code, diff = extract_code("no code here, just prose { not python }")
    assert code == ""
    assert diff.method == "none"


def test_apply_diff_raises_on_missing_search():
    with pytest.raises(ValueError):
        _apply_diff(PRIOR, [("absent block", "x")])
