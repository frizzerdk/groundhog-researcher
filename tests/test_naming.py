"""Attempt naming — slugify + direction-title derivation (producer side)."""

from groundhog.utils.direction import slugify, workspace_name, write_direction


def test_slugify_normalizes():
    assert slugify("CNN architecture") == "cnn-architecture"
    assert slugify("rollout-based search") == "rollout-based-search"
    assert slugify("already-a-slug") == "already-a-slug"
    assert slugify('  "Quoted" Name!! ') == "quoted-name"
    assert slugify("") == ""


def test_slugify_caps_word_count():
    assert slugify("one two three four five six seven", max_words=3) == "one-two-three"


def test_workspace_name_from_direction_title(tmp_path):
    write_direction(tmp_path, "CNN architecture\nwith heavy dropout")
    # Uses the first direction line, slugified.
    assert workspace_name(tmp_path) == "cnn-architecture"


def test_workspace_name_explicit_wins(tmp_path):
    write_direction(tmp_path, "CNN architecture")
    assert workspace_name(tmp_path, explicit="planned-name") == "planned-name"


def test_workspace_name_empty_without_direction(tmp_path):
    assert workspace_name(tmp_path) == ""
