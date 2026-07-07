"""Searchable knowledge base over a run — corpus walk + rebuildable semantic index.

Two tiers:

Tier 1 (``iter_corpus``): the knowledge corpus of a run, read through the
``AttemptHistory`` interface so every backend works — run-root ``learnings.md``
and ``insights.md``, plus each attempt's ``core_direction.md``,
``attemptlog.md``, and ``work/learnings.md`` where present. The lexical
``search-attempts`` agent tool scans this directly.

Tier 2 (``SemanticIndex``): a DERIVED cache of embedding vectors over the same
corpus, stored as JSONL under ``.groundhog-cache/``. It is never authoritative:
delete it and ``rebuild()`` re-derives an identical index from the store.

Embedders: anything with a ``name`` attribute and
``embed(texts) -> list of vectors`` (dense ``list[float]`` or sparse
``dict[token, weight]``). The built-in fallback is :class:`TfidfEmbedder` —
pure-python TF-IDF over word tokens, no dependencies. To plug in a real
embedding backend, wrap its API client yourself::

    class MyEmbedder:
        name = "my-model-v1"
        def embed(self, texts):
            return client.embed(texts)   # list[list[float]]

    index = SemanticIndex(root, history, embedder=MyEmbedder())
    index.rebuild()

The cache records the embedder name; loading with a different embedder fails
so stale vectors are never scored against a mismatched query embedding —
rebuild instead. Corpus-dependent embedders may also expose ``fit(texts)``
(called during rebuild) and ``state()`` / ``load_state(dict)`` (persisted in
the cache header) — the TF-IDF fallback uses these for its idf table.
"""

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

CACHE_DIR = ".groundhog-cache"
CACHE_FILE = "semantic.jsonl"

SCOPES = ("all", "learnings", "directions", "logs", "insights")

_ROOT_FILES = {"learnings.md": "learnings", "insights.md": "insights"}
_ATTEMPT_FILES = (
    ("core_direction.md", "directions"),
    ("work/core_direction.md", "directions"),
    ("attemptlog.md", "logs"),
    ("work/learnings.md", "learnings"),
)


@dataclass
class CorpusDoc:
    """One knowledge file. ``attempt`` is None for run-root files."""
    attempt: Optional[str]
    file: str
    text: str


def iter_corpus(root, history, scope: str = "all") -> Iterator[CorpusDoc]:
    """Yield the run's knowledge files, attempt-stamped.

    ``root`` is the run directory (may be None); ``history`` an
    AttemptHistory (may be None). Reads attempt files via the Attempt
    interface, never the on-disk layout.
    """
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r} (use {'|'.join(SCOPES)})")
    if root is not None:
        root = Path(root)
        for name, file_scope in _ROOT_FILES.items():
            if scope not in ("all", file_scope):
                continue
            path = root / name
            if path.exists():
                try:
                    text = path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                if text.strip():
                    yield CorpusDoc(attempt=None, file=name, text=text)
    if history is None:
        return
    for attempt in history.list(only_done=False):
        seen_direction = False
        for name, file_scope in _ATTEMPT_FILES:
            if scope not in ("all", file_scope):
                continue
            if file_scope == "directions" and seen_direction:
                continue
            text = attempt.read_file(name)
            if text and text.strip():
                if file_scope == "directions":
                    seen_direction = True
                yield CorpusDoc(attempt=attempt.id, file=name, text=text)


_ENTRY_SEP = re.compile(r"\n-{3,}\s*\n")


def chunk_corpus(docs) -> List[CorpusDoc]:
    """Chunk docs for indexing: per-file, except run-root learnings/insights,
    which split on their documented ``---`` entry separator (they grow
    unboundedly over a run; one giant chunk would drown every query)."""
    chunks = []
    for doc in docs:
        if doc.attempt is None:
            parts = [p for p in _ENTRY_SEP.split(doc.text) if p.strip()]
            if len(parts) > 1:
                for i, part in enumerate(parts, start=1):
                    chunks.append(CorpusDoc(doc.attempt, f"{doc.file}#{i}", part))
                continue
        chunks.append(doc)
    return chunks


_TOKEN = re.compile(r"[a-z0-9_]+")


def _tokens(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


class TfidfEmbedder:
    """Pure-python TF-IDF fallback embedder. Vectors are sparse dicts."""

    name = "tfidf"

    def __init__(self):
        self.idf = {}

    def fit(self, texts: Sequence[str]) -> None:
        n = len(texts)
        df = Counter()
        for t in texts:
            df.update(set(_tokens(t)))
        self.idf = {tok: math.log((1 + n) / (1 + c)) + 1.0 for tok, c in df.items()}

    def embed(self, texts: Sequence[str]) -> List[dict]:
        out = []
        for t in texts:
            tf = Counter(_tokens(t))
            total = sum(tf.values()) or 1
            vec = {tok: (cnt / total) * self.idf[tok]
                   for tok, cnt in tf.items() if tok in self.idf}
            out.append(vec)
        return out

    def state(self) -> dict:
        return {"idf": self.idf}

    def load_state(self, state: dict) -> None:
        self.idf = dict(state.get("idf", {}))


def _cosine(a, b) -> float:
    if isinstance(a, dict):
        if len(b) < len(a):
            a, b = b, a
        dot = sum(v * b.get(k, 0.0) for k, v in a.items())
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
    else:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


@dataclass
class SemanticHit:
    attempt: Optional[str]
    file: str
    score: float
    preview: str


_PREVIEW_CHARS = 200


def _preview(text: str) -> str:
    return " ".join(text.split())[:_PREVIEW_CHARS]


class SemanticIndex:
    """Rebuildable embedding index over the run's knowledge corpus.

    A derived cache: ``rebuild()`` re-derives everything from the store, so
    the cache file is always safe to delete. See the module docstring for
    the embedder contract and how to plug in a real embedding backend.
    """

    def __init__(self, root, history=None, embedder=None, cache_path=None):
        self.root = Path(root)
        self.history = history
        self.embedder = embedder or TfidfEmbedder()
        self.cache_path = (Path(cache_path) if cache_path
                           else self.root / CACHE_DIR / CACHE_FILE)
        self._chunks = None

    def exists(self) -> bool:
        return self.cache_path.exists()

    def rebuild(self) -> int:
        """Re-derive the index from the store. Returns the chunk count."""
        chunks = chunk_corpus(iter_corpus(self.root, self.history))
        texts = [c.text for c in chunks]
        if hasattr(self.embedder, "fit"):
            self.embedder.fit(texts)
        vectors = self.embedder.embed(texts) if texts else []
        meta = {"kind": "meta", "embedder": self.embedder.name}
        if hasattr(self.embedder, "state"):
            meta["state"] = self.embedder.state()
        lines = [json.dumps(meta)]
        records = []
        for chunk, vec in zip(chunks, vectors):
            rec = {"attempt": chunk.attempt, "file": chunk.file,
                   "preview": _preview(chunk.text), "vector": vec}
            records.append(rec)
            lines.append(json.dumps({"kind": "chunk", **rec}))
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._chunks = records
        return len(records)

    def load(self) -> bool:
        """Load the cache. False when absent; ValueError on embedder mismatch."""
        if not self.exists():
            return False
        records = []
        meta = None
        try:
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("kind") == "meta":
                    meta = rec
                else:
                    records.append(rec)
        except (OSError, ValueError):
            return False
        if meta is None:
            return False
        if meta.get("embedder") != self.embedder.name:
            raise ValueError(
                f"semantic cache was built with embedder "
                f"{meta.get('embedder')!r}, not {self.embedder.name!r} — "
                f"rebuild() to re-derive it"
            )
        if hasattr(self.embedder, "load_state"):
            self.embedder.load_state(meta.get("state", {}))
        self._chunks = records
        return True

    def search(self, query: str, k: int = 5) -> List[SemanticHit]:
        """Rank corpus chunks by cosine similarity to ``query``."""
        if self._chunks is None:
            if not self.load():
                self.rebuild()
        qv = self.embedder.embed([query])[0]
        scored = [
            SemanticHit(attempt=rec["attempt"], file=rec["file"],
                        score=_cosine(qv, rec["vector"]),
                        preview=rec["preview"])
            for rec in self._chunks
        ]
        scored.sort(key=lambda h: (-h.score, h.attempt or "", h.file))
        return scored[:k]
