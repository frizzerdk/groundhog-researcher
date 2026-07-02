"""groundhog skills install: packaged skills land in <run>/.claude/skills/.

Idempotent, overwrite-on-update, never touches non-groundhog skills, and
every init* wires it in — skill drift dies with the package version.
"""


from groundhog.cli import SKILLS_DIR, init, install_skills, skills_group

EXPECTED = {
    "groundhog-interface",
    "groundhog-in-attempt",
    "groundhog-fresh",
    "groundhog-improve",
    "groundhog-iterate",
    "groundhog-orchestrate",
}


def test_packaged_skills_present():
    names = {p.name for p in SKILLS_DIR.iterdir() if p.is_dir()}
    assert EXPECTED <= names
    for name in EXPECTED:
        assert (SKILLS_DIR / name / "SKILL.md").exists()


def test_install_copies_all_skills(tmp_path):
    rc = install_skills(tmp_path)
    assert rc == 0
    dest = tmp_path / ".claude" / "skills"
    installed = {p.name for p in dest.iterdir() if p.is_dir()}
    assert EXPECTED <= installed
    for name in EXPECTED:
        assert (dest / name / "SKILL.md").exists()


def test_install_is_idempotent_and_overwrites_drift(tmp_path):
    install_skills(tmp_path)
    drifted = tmp_path / ".claude" / "skills" / "groundhog-fresh" / "SKILL.md"
    drifted.write_text("stale local edit", encoding="utf-8")

    rc = install_skills(tmp_path)
    assert rc == 0
    packaged = (SKILLS_DIR / "groundhog-fresh" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert drifted.read_text(encoding="utf-8") == packaged


def test_install_never_touches_foreign_skills(tmp_path):
    foreign = tmp_path / ".claude" / "skills" / "my-own-skill"
    foreign.mkdir(parents=True)
    (foreign / "SKILL.md").write_text("mine", encoding="utf-8")

    install_skills(tmp_path)
    assert (foreign / "SKILL.md").read_text(encoding="utf-8") == "mine"


def test_skills_group_dispatch(tmp_path, capsys):
    rc = skills_group(["install", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Installed" in out
    assert (tmp_path / ".claude" / "skills" / "groundhog-interface").exists()

    assert skills_group([]) == 0  # help
    assert skills_group(["nope"]) == 1


def test_init_installs_skills(tmp_path, capsys):
    target = tmp_path / "fresh_run"
    rc = init("init-mock", str(target), script_only=True)
    assert rc == 0
    dest = target / ".claude" / "skills"
    installed = {p.name for p in dest.iterdir() if p.is_dir()}
    assert EXPECTED <= installed
