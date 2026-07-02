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


def test_direction_title_strips_redundant_core_direction_label():
    """Agents often title the file '# Core Direction: X' — the label is
    redundant (the file IS the core direction) and used to leak into every
    slug (core-direction-data-augmentation-...), status line, and family
    title (2026-07-02 morning review note)."""
    from groundhog.utils.direction import direction_title, slugify

    assert direction_title("# Core Direction: Data Augmentation + Random Forest") == \
        "Data Augmentation + Random Forest"
    assert direction_title("Core direction - prototype matching") == "prototype matching"
    assert direction_title("CORE_DIRECTION: knn ensemble") == "knn ensemble"
    # A body whose first line is ONLY the label falls through to the content.
    assert direction_title("# Core Direction\nprototype matching\n") == "prototype matching"
    # No label -> unchanged; and 'directional' content is not mangled.
    assert direction_title("Directional gradient descent") == "Directional gradient descent"
    assert slugify(direction_title("# Core Direction: Data Augmentation + Random Forest")) == \
        "data-augmentation-random-forest"
