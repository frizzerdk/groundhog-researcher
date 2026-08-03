"""Knowledge-base search — tier 1 lexical tool + tier 2 semantic index.

Tier 1: the ``search-attempts`` default agent tool scans run-root
learnings/insights plus per-attempt directions, logs, and work-learnings
through the AttemptHistory interface, returning attempt-stamped hits.

Tier 2: ``SemanticIndex`` is a derived cache — rebuildable from the store,
identical after delete + rebuild, zero hard dependencies (TF-IDF fallback).
"""

import json
from pathlib import Path

import pytest

from groundhog.base.toolkit import Toolkit
from groundhog.base.types import EvaluationResult, StageResult
from groundhog.histories.folder import FolderAttemptHistory
from groundhog.utils.direction import write_direction
from groundhog.utils.results import write_result
from groundhog.utils.semantic_index import (
    SemanticIndex, TfidfEmbedder, chunk_corpus, iter_corpus,
)
from groundhog.agents.tools import build_default_agent_tools, search_attempts


def _commit(history, *, parent=None, direction=None, attemptlog=None,
            work_learnings=None):
    ws = history.workspace(parent=parent)
    (ws.path / "solution.py").write_text("def solve(): return 1", encoding="utf-8")
    if direction:
        write_direction(ws.path, direction)
    if attemptlog:
        (ws.path / "attemptlog.md").write_text(attemptlog, encoding="utf-8")
    if work_learnings:
        (ws.path / "work").mkdir(exist_ok=True)
        (ws.path / "work" / "learnings.md").write_text(work_learnings, encoding="utf-8")
    write_result(ws.path, EvaluationResult(
        stages={"eval": StageResult(metrics={"score": 1.0})}, completed=True))
    return ws.commit(success=True)


def _synthetic_run(root, history):
    Path(root, "learnings.md").write_text(
        "Batch norm before activation helps.\n"
        "---\n"
        "Dropout above 0.5 collapsed training.\n",
        encoding="utf-8")
    Path(root, "insights.md").write_text(
        "Cross-attempt insight: augmentation beats depth here.\n",
        encoding="utf-8")
    a1 = _commit(history,
                 direction="Use a convolutional network with augmentation",
                 attemptlog="ran eval, augmentation pipeline crashed once",
                 work_learnings="rotation augmentation gained 2 points")
    a2 = _commit(history, parent=a1.id,
                 direction="Wider layers with heavy dropout regularization",
                 attemptlog="dropout 0.7 diverged; lowered to 0.3")
    return a1, a2


@pytest.fixture
def run(tmp_path):
    history = FolderAttemptHistory(tmp_path)
    a1, a2 = _synthetic_run(tmp_path, history)
    toolkit = Toolkit(task=None, history=history, path=tmp_path)
    return toolkit, a1, a2


# === Tier 1: lexical search-attempts tool ===

def test_lexical_search_stamps_hits_with_attempt_and_file(run):
    toolkit, a1, a2 = run
    out = search_attempts(toolkit, "augmentation")
    assert f"attempt_{a1.id} core_direction.md:1:" in out
    assert f"attempt_{a1.id} attemptlog.md:1:" in out
    assert f"attempt_{a1.id} work/learnings.md:1:" in out
    assert "run-root insights.md:1:" in out


def test_lexical_search_scopes_narrow_the_corpus(run):
    toolkit, a1, a2 = run
    directions = search_attempts(toolkit, "dropout", scope="directions")
    assert "core_direction.md" in directions
    assert "attemptlog.md" not in directions

    logs = search_attempts(toolkit, "dropout", scope="logs")
    assert "attemptlog.md" in logs
    assert "core_direction.md" not in logs

    learnings = search_attempts(toolkit, "dropout", scope="learnings")
    assert "run-root learnings.md" in learnings

    insights = search_attempts(toolkit, "augmentation", scope="insights")
    assert insights.splitlines()[1:] == [
        "run-root insights.md:1: Cross-attempt insight: augmentation beats depth here."
    ]


def test_lexical_search_accepts_regex_and_bad_regex_falls_back(run):
    toolkit, _, a2 = run
    out = search_attempts(toolkit, r"dropout 0\.[0-9]")
    assert f"attempt_{a2.id} attemptlog.md" in out
    # Invalid regex degrades to a literal scan instead of raising.
    assert search_attempts(toolkit, "dropout (").startswith("(no hits")


def test_lexical_search_caps_results_and_reports_misses(run):
    toolkit, _, _ = run
    out = search_attempts(toolkit, "a", max_results=2)
    assert out.startswith("2 hit(s)")
    assert len(out.splitlines()) == 3
    assert search_attempts(toolkit, "zzz-not-present").startswith("(no hits")
    assert "unknown scope" in search_attempts(toolkit, "a", scope="bogus")


def test_search_attempts_is_a_default_agent_tool(run):
    toolkit, a1, _ = run
    tools = {t.name: t for t in build_default_agent_tools(toolkit)}
    assert "search-attempts" in tools
    result = tools["search-attempts"].execute(query="augmentation")
    assert result.success
    assert f"attempt_{a1.id}" in result.output


def test_lexical_search_works_on_both_history_backends(history_factory, tmp_path):
    history = history_factory()
    a1, a2 = _synthetic_run(tmp_path, history)
    toolkit = Toolkit(task=None, history=history, path=tmp_path)
    out = search_attempts(toolkit, "dropout")
    assert f"attempt_{a2.id} core_direction.md:1:" in out
    assert "run-root learnings.md" in out


# === Tier 2: semantic index (fallback vectorizer) ===

def test_corpus_chunks_split_run_root_entries(run):
    toolkit, _, _ = run
    chunks = chunk_corpus(iter_corpus(toolkit.path, toolkit.history))
    files = [c.file for c in chunks if c.attempt is None]
    assert "learnings.md#1" in files and "learnings.md#2" in files


def test_index_build_and_search_ranks_relevant_chunk_first(run):
    toolkit, _, a2 = run
    index = SemanticIndex(toolkit.path, toolkit.history)
    n = index.rebuild()
    assert n >= 6
    assert index.cache_path == toolkit.path / ".groundhog-cache" / "semantic.jsonl"
    hits = index.search("dropout regularization layers", k=3)
    assert hits[0].attempt == a2.id
    assert hits[0].file == "core_direction.md"
    assert hits[0].score > 0
    assert "dropout" in hits[0].preview.lower()


def test_index_is_rebuildable_delete_cache_identical_results(run):
    toolkit, _, _ = run
    index = SemanticIndex(toolkit.path, toolkit.history)
    index.rebuild()
    before = [(h.attempt, h.file, round(h.score, 12), h.preview)
              for h in index.search("augmentation crashed", k=5)]

    index.cache_path.unlink()
    fresh = SemanticIndex(toolkit.path, toolkit.history)
    assert not fresh.exists()
    fresh.rebuild()
    after = [(h.attempt, h.file, round(h.score, 12), h.preview)
             for h in fresh.search("augmentation crashed", k=5)]
    assert before == after


def test_index_loads_from_cache_without_store_access(run):
    toolkit, _, _ = run
    SemanticIndex(toolkit.path, toolkit.history).rebuild()
    cold = SemanticIndex(toolkit.path, history=None)
    assert cold.load()
    assert cold.search("dropout", k=1)[0].score > 0


def test_index_refuses_mismatched_embedder(run):
    toolkit, _, _ = run
    SemanticIndex(toolkit.path, toolkit.history).rebuild()

    class Other(TfidfEmbedder):
        name = "other-model"

    with pytest.raises(ValueError, match="rebuild"):
        SemanticIndex(toolkit.path, toolkit.history, embedder=Other()).load()


def test_lexical_search_uses_semantic_rank_when_index_exists(run):
    toolkit, _, a2 = run
    out = search_attempts(toolkit, "dropout")
    assert "semantic rank" not in out.splitlines()[0]

    SemanticIndex(toolkit.path, toolkit.history).rebuild()
    ranked = search_attempts(toolkit, "dropout regularization")
    header, first = ranked.splitlines()[:2]
    assert "semantic rank" in header
    assert first.startswith(f"attempt_{a2.id} core_direction.md")


def test_corrupt_cache_falls_back_to_lexical_order(run):
    toolkit, _, _ = run
    index = SemanticIndex(toolkit.path, toolkit.history)
    index.rebuild()
    index.cache_path.write_text("{not json", encoding="utf-8")
    out = search_attempts(toolkit, "dropout")
    assert "semantic rank" not in out.splitlines()[0]
    assert "dropout" in out.lower()


# === Staleness fingerprint + explicit rebuild ===

def test_stale_index_skips_semantic_rank_until_rebuilt(run):
    """An index built before new commits must not claim semantic order (it
    would silently sort post-rebuild attempts last). The corpus fingerprint
    detects the mismatch; an explicit rebuild re-enables ranking."""
    toolkit, _, _ = run
    SemanticIndex(toolkit.path, toolkit.history).rebuild()
    assert "semantic rank" in search_attempts(toolkit, "dropout").splitlines()[0]

    a3 = _commit(toolkit.history,
                 direction="Ensemble of small models with bagging")
    out = search_attempts(toolkit, "dropout")
    assert "semantic rank" not in out.splitlines()[0]

    tools = {t.name: t for t in build_default_agent_tools(toolkit)}
    assert "rebuild-kb-index" in tools
    result = tools["rebuild-kb-index"].execute()
    assert result.success
    assert "rebuilt" in result.output

    ranked = search_attempts(toolkit, "bagging")
    assert "semantic rank" in ranked.splitlines()[0]
    assert f"attempt_{a3.id}" in ranked


def test_custom_toolkit_embedder_is_used_by_search_and_rebuild(run):
    toolkit, _, _ = run

    class Counting(TfidfEmbedder):
        name = "counting-v1"
        calls = []

        def embed(self, texts):
            Counting.calls.append(len(texts))
            return super().embed(texts)

    toolkit.embedder = Counting()
    tools = {t.name: t for t in build_default_agent_tools(toolkit)}
    assert tools["rebuild-kb-index"].execute().success
    assert Counting.calls  # rebuild embedded through the custom embedder

    # The shipped search tool loads the custom-embedder cache (the TF-IDF
    # default would refuse it as a mismatch).
    out = search_attempts(toolkit, "dropout regularization")
    assert "semantic rank" in out.splitlines()[0]


def test_agent_supplied_regex_is_length_capped(run):
    toolkit, _, _ = run
    huge = "(a+)+" * 5000
    out = search_attempts(toolkit, huge)
    assert out.startswith("(no hits")  # capped + compiled, no crash
    long_literal = "dropout" + " " * 600
    assert search_attempts(toolkit, long_literal) is not None
