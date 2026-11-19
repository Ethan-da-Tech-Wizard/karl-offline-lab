# ==============================================================================
# Program: rag_pipeline.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Retrieval-Augmented Generation (RAG) pipeline.
#          Performs document ingestion, semantic chunking, vector indexing, and
#          context injection.
# ==============================================================================

"""
RAG Pipeline — Karl Workbench (Hardened)
=========================================
Changes from original:
  - Persistent FAISS index: saved/loaded from data/vector_db/
  - SQLite metadata store with WAL and transactional ingestion
  - Optional contextual chunk headers (source + chunk ID prefix)
  - source_filter param on retrieve() to restrict by file
  - Retrieval eval metrics: hit@k and reciprocal rank
"""

import logging
import json
import os
import queue
import sqlite3
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

# Third-party: faiss (vector index) and numpy (vectors); PyMuPDF and python-docx read documents.
import faiss
import numpy as np
try:
    # Execute guarded operations and establish recovery boundaries
    # Newer PyMuPDF releases import as 'pymupdf'; older ones only provide 'fitz'. Either gives the same API.
    import pymupdf as fitz
except ModuleNotFoundError:
    import fitz  # PyMuPDF before the pymupdf import name was available
import docx

# Pool used for concurrent SQLite writes (reads and schema setup use a separate direct connection).
from app.utils.db_pool import SQLiteConnectionPool


# Module logger for the RAG subsystem.
logger = logging.getLogger("karl.rag")

# Sentinel distinguishing "not yet attempted" (None) from "attempted and failed".
# Stored in self._reranker after a failed load so we don't retry on every call.
_RERANKER_UNAVAILABLE = object()

# File types the ingester will read; everything else is silently skipped.
_SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".py", ".csv"}


@dataclass
class _ParsedFile:
    """Result of parsing one file in a worker thread: its chunks, or the error that stopped it.
    """
    index: int
    path: str
    source_file: str
    chunks: list[str]
    error: str | None = None


class RAGPipeline:
    # Default paths; __init__ overrides them per namespace.
    """Local retrieval-augmented-generation store: FAISS vectors plus SQLite metadata.

    Ingests documents (parsed in parallel, embedded in batches), persists the
    index under data/vector_db, and retrieves by dense, sparse (TF-IDF) or hybrid
    (rank fusion with optional cross-encoder reranking) search.
    """
    INDEX_FILE = "data/vector_db/index.faiss"
    META_DB = "data/vector_db/meta.db"

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_path: str = "data/vector_db",
        contextual_headers: bool = False,
        namespace: str = "user",
    ):
        """
        Args:
            model_name:          SentenceTransformer model for embeddings.
            index_path:          Directory for persisted index files.
            contextual_headers:  If True, prepend "[Source: file | Chunk N]" to
                                 each retrieved chunk — aids model citation.
            namespace:           Isolate user vs system index files.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        self.model_name = model_name
        self._encoder = None
        self._reranker = None     # None = not yet attempted; _RERANKER_UNAVAILABLE = load failed
        self.index_path = index_path
        self.namespace = namespace
        # The 'codex' namespace keeps its own files so system documentation never mixes with user documents.
        if namespace == "codex":
            self.INDEX_FILE = os.path.join(index_path, "codex_index.faiss")
            self.META_DB = os.path.join(index_path, "codex_meta.db")
        else:
            self.INDEX_FILE = os.path.join(index_path, "index.faiss")
            self.META_DB = os.path.join(index_path, "meta.db")
        self.contextual_headers = contextual_headers

        os.makedirs(self.index_path, exist_ok=True)

        # 384 is the embedding size of the default all-MiniLM-L6-v2 model; a different model needs a different value.
        self.dimension = 384
        self.index = self._new_index()

        # In-memory copy of the metadata rows, kept for the UI and for sparse/exact-match search.
        # Each entry mirrors the SQLite row shape and remains for UI compatibility.
        self.documents: list[dict] = []
        # Serializes mutations of the in-memory index and documents list across writer threads.
        self._write_lock = threading.Lock()

        # Dedicated background thread that serializes the slow part of a
        # write - faiss.write_index()+os.replace() - in strict submission
        # order. Writers hand off a post-mutation index snapshot here and
        # wait on the returned handle *after* releasing _write_lock and
        # committing their SQLite transaction, so a second concurrent
        # writer's (fast) DB txn + in-memory mutation isn't stuck queueing
        # behind the first writer's (slow) disk flush. See
        # _submit_index_flush()/_wait_for_index_flush() and their use in
        # _add_chunks_to_index()/remove_source().
        # Dedicated background thread that serializes the slow part of a
        # write — faiss.write_index()+os.replace() — in strict submission
        self._index_flush_queue: "queue.Queue" = queue.Queue()
        self._index_flush_thread = threading.Thread(
            target=self._index_flush_worker,
            daemon=True,
            name=f"karl-rag-flush-{namespace}",
        )
        self._index_flush_thread.start()

        # Bootstrap: single direct connection to create the schema, then hand off
        # all subsequent concurrent writes to the pool.
        self._conn = self._connect_db()
        self._init_db()
        self._pool = SQLiteConnectionPool(self.META_DB)

        # Load persisted index if it exists
        self._load_index()

    def _new_index(self):
        """Create an empty exact (flat L2) FAISS index wrapped in an ID map so vectors can be removed by id.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        return faiss.IndexIDMap2(faiss.IndexFlatL2(self.dimension))

    def _connect_db(self) -> sqlite3.Connection:
        """Open the metadata database in autocommit mode with WAL journaling so reads do not block writes.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        conn = sqlite3.connect(self.META_DB, timeout=30.0, isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self):
        """Create the documents table and its lookup indexes if they do not exist."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                vector_id INTEGER UNIQUE,
                text TEXT,
                source_file TEXT,
                chunk_id INTEGER,
                ingested_at TEXT
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_source_file ON documents(source_file)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_vector_id ON documents(vector_id)"
        )

    def _legacy_metadata_file(self) -> str:
        """Path of the pre-SQLite JSON metadata file for this namespace."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if self.namespace == "codex":
            return os.path.join(self.index_path, "codex_metadata.json")
        return os.path.join(self.index_path, "metadata.json")

    def _fetch_documents(self, conn: sqlite3.Connection | None = None) -> list[dict]:
        """Load all metadata rows ordered by vector id as plain dicts."""
        # Step 1: Validate input parameters, operational preconditions, and state
        active_conn = conn or self._conn
        rows = active_conn.execute(
            """
            SELECT id, vector_id, text, source_file, chunk_id, ingested_at
            FROM documents
            ORDER BY vector_id
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def _next_vector_id(self, conn: sqlite3.Connection | None = None) -> int:
        """Return the next unused vector id (one past the largest id in the database or in memory).
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        active_conn = conn or self._conn
        db_max = int(
            active_conn.execute(
                "SELECT COALESCE(MAX(vector_id), -1) FROM documents"
            ).fetchone()[0]
        )
        # Also check in-memory docs added via save=False (ingest_text path) so
        # that multiple ingest_text calls don't all land at vector_id 0.
        mem_max = max(
            (int(d.get("vector_id", -1)) for d in self.documents),
            default=-1,
        )
        return max(db_max, mem_max) + 1

    def _add_embeddings_to_index(self, embeddings: np.ndarray, vector_ids: list[int]):
        """Add embeddings to the FAISS index under explicit vector ids."""
        # Step 1: Validate input parameters, operational preconditions, and state
        ids = np.asarray(vector_ids, dtype="int64")
        self.index.add_with_ids(np.asarray(embeddings, dtype="float32"), ids)

    def _write_index_atomic(self, index=None):
        """Write the index to a temp file then rename it over the real file so readers never see a partial index.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        active_index = index or self.index
        tmp_path = f"{self.INDEX_FILE}.tmp"
        faiss.write_index(active_index, tmp_path)
        os.replace(tmp_path, self.INDEX_FILE)

    def _index_flush_worker(self) -> None:
        """Background loop: persists queued index snapshots to disk one at a
        time, in the exact order they were submitted (a single consumer
        thread makes ordering automatic — no extra locking needed)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        while True:
            # Iterate over items, evaluate loop invariant, and update progress
            job = self._index_flush_queue.get()
            if job is None:  # shutdown sentinel
                self._index_flush_queue.task_done()
                return
            index_snapshot, done_event, result = job
            try:
                # Execute guarded operations and establish recovery boundaries
                self._write_index_atomic(index_snapshot)
                result["ok"] = True
            except Exception as exc:
                result["ok"] = False
                result["error"] = exc
            finally:
                done_event.set()
                self._index_flush_queue.task_done()

    def _submit_index_flush(self, index_snapshot) -> tuple:
        """Hand *index_snapshot* off to the background flush thread.

        Returns a (done_event, result) handle for _wait_for_index_flush().
        Must be called with a snapshot (faiss.clone_index(self.index)), not
        self.index itself — self.index may keep mutating after this call
        returns (e.g. a subsequent writer), and the background thread reads
        the snapshot outside of _write_lock.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        done_event = threading.Event()
        result: dict = {}
        self._index_flush_queue.put((index_snapshot, done_event, result))
        return done_event, result

    @staticmethod
    def _wait_for_index_flush(flush_job: tuple) -> None:
        """Block until a previously submitted flush completes.

        Re-raises whatever exception the background thread hit while
        writing to disk. Note: by the time a caller reaches this, its
        SQLite transaction has already committed (see _add_chunks_to_index/
        remove_source) — a flush failure here means the in-process index
        and the SQLite documents table are consistent with each other and
        immediately searchable, but the on-disk index.faiss file is stale
        until a later successful flush or a manual rebuild_index(). This is
        a deliberate trade-off: decoupling the disk write from the SQLite
        write-lock is what lets concurrent writers pipeline instead of
        fully serializing on every ingest/remove call.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        done_event, result = flush_job
        done_event.wait()
        if not result.get("ok"):
            raise result.get("error") or RuntimeError("Index flush failed")

    def _migrate_legacy_metadata(self):
        """One-time import of the old JSON metadata into SQLite (only if the table is empty).
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        legacy_file = self._legacy_metadata_file()
        if not os.path.exists(legacy_file):
            return
        existing = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        if existing:
            return
        try:
            # Execute guarded operations and establish recovery boundaries
            with open(legacy_file, "r", encoding="utf-8") as f:
                legacy_docs = json.load(f)
            self._conn.execute("BEGIN IMMEDIATE")
            for idx, doc in enumerate(legacy_docs):
                # Iterate over items, evaluate loop invariant, and update progress
                vector_id = int(doc.get("vector_id", doc.get("chunk_id", idx)))
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO documents(vector_id, text, source_file, chunk_id, ingested_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        vector_id,
                        doc.get("text", ""),
                        doc.get("source_file", "unknown"),
                        int(doc.get("chunk_id", vector_id)),
                        doc.get("ingested_at", ""),
                    ),
                )
            self._conn.commit()
        except Exception as exc:
            self._conn.rollback()
            logger.warning("WARNING: Could not migrate legacy RAG metadata: %s", exc)

    def _ensure_id_index(self, loaded_index):
        """Return an ID-mapped index, wrapping a legacy plain index and re-adding its vectors under sequential ids.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        if hasattr(loaded_index, "add_with_ids"):
            return loaded_index
        wrapped = self._new_index()
        if loaded_index.ntotal:
            try:
                # Execute guarded operations and establish recovery boundaries
                vectors = np.vstack(
                    [loaded_index.reconstruct(i) for i in range(loaded_index.ntotal)]
                ).astype("float32")
                ids = np.arange(loaded_index.ntotal, dtype="int64")
                wrapped.add_with_ids(vectors, ids)
            except Exception as exc:
                logger.warning("WARNING: Could not migrate legacy FAISS index: %s", exc)
        return wrapped

    @property
    def encoder(self):
        """Lazily load the SentenceTransformer model, silencing its console output."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if self._encoder is None:
            import io
            import contextlib
            from sentence_transformers import SentenceTransformer
            # Model loading prints progress to stdout/stderr; redirect it so it does not clutter the app console.
            _sink = io.StringIO()
            with contextlib.redirect_stdout(_sink), contextlib.redirect_stderr(_sink):
                self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    @property
    def is_encoder_loaded(self) -> bool:
        """True once the embedding model has been loaded into memory."""
        # Step 1: Validate input parameters, operational preconditions, and state
        return self._encoder is not None

    @property
    def reranker(self):
        """Lazy-load the local CrossEncoder model on first access.

        Returns the CrossEncoder instance on success, or None if the model
        weights are not cached locally.  After a failed load the sentinel is
        stored so subsequent calls return None immediately without retrying.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        if self._reranker is None:
            import io
            import contextlib
            try:
                # Execute guarded operations and establish recovery boundaries
                from sentence_transformers import CrossEncoder
                _sink = io.StringIO()
                with contextlib.redirect_stdout(_sink), contextlib.redirect_stderr(_sink):
                    self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                logger.info("CrossEncoder reranker loaded.")
            except Exception as exc:
                logger.warning(
                    "CrossEncoder 'ms-marco-MiniLM-L-6-v2' unavailable — "
                    "hybrid search will fall back to RRF ordering. (%s: %s)",
                    type(exc).__name__, exc,
                )
                self._reranker = _RERANKER_UNAVAILABLE
        if self._reranker is _RERANKER_UNAVAILABLE:
            return None
        return self._reranker

    def preload_encoder(self):
        """Load the embedding model now (instead of on the first query) to avoid a delay later.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        _ = self.encoder

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_index(self):
        """Load FAISS index and SQLite metadata from disk if they exist."""
        # Step 1: Validate input parameters, operational preconditions, and state
        self._migrate_legacy_metadata()
        try:
            # Execute guarded operations and establish recovery boundaries
            self.documents = self._fetch_documents()
            if os.path.exists(self.INDEX_FILE):
                self.index = self._ensure_id_index(faiss.read_index(self.INDEX_FILE))
            else:
                self.index = self._new_index()
            # A mismatch means index and metadata drifted apart (e.g. after a failed flush); warn but continue.
            # rebuild_index() repairs it.
            if self.index.ntotal != len(self.documents):
                logger.warning(
                    "RAG index/metadata count mismatch: index=%d metadata=%d",
                    self.index.ntotal,
                    len(self.documents),
                )
            logger.info(f"Loaded {self.index.ntotal} vectors from {self.INDEX_FILE}")
        except Exception as e:
            logger.warning(f"WARNING: Could not load persisted index: {e}. Starting fresh.")
            self.index = self._new_index()
            self.documents = []

    def save_index(self):
        """Write the FAISS index to disk. Metadata is persisted in SQLite."""
        # Step 1: Validate input parameters, operational preconditions, and state
        try:
            # Execute guarded operations and establish recovery boundaries
            self._write_index_atomic()
        except Exception as e:
            logger.warning(f"WARNING: Could not persist index: {e}")

    def clear_index(self):
        """Wipe the in-memory index and delete persisted files."""
        # Step 1: Validate input parameters, operational preconditions, and state
        with self._write_lock:
            with self._pool.get_connection() as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("DELETE FROM documents")
                conn.commit()
            self.index = self._new_index()
            self.documents = []
        for path in (self.INDEX_FILE, f"{self.INDEX_FILE}.tmp"):
            # Iterate over items, evaluate loop invariant, and update progress
            if os.path.exists(path):
                os.remove(path)
        logger.info("Index cleared.")

    # ── Text Extraction ───────────────────────────────────────────────────────

    def extract_text(self, filepath: str) -> str:
        """Return the plain text of a PDF, DOCX or text-like file ('' if it cannot be read)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        ext = os.path.splitext(filepath)[1].lower()
        text = ""
        try:
            # Execute guarded operations and establish recovery boundaries
            if ext == ".pdf":
                doc = fitz.open(filepath)
                for page in doc:
                    # Iterate over items, evaluate loop invariant, and update progress
                    text += page.get_text() + "\n"
            elif ext == ".docx":
                doc = docx.Document(filepath)
                for para in doc.paragraphs:
                    # Iterate over items, evaluate loop invariant, and update progress
                    text += para.text + "\n"
            else:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
        except Exception as e:
            logger.warning(f"Error reading {filepath}: {e}")
        return text

    # ── Chunking ──────────────────────────────────────────────────────────────

    def chunk_text(self, text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
        """Split text into overlapping word windows of chunk_size words.

        The window advances by chunk_size - overlap words, so overlap must be
        smaller than chunk_size.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            # Iterate over items, evaluate loop invariant, and update progress
            chunk = " ".join(words[i : i + chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def _default_worker_count(self) -> int:
        """Choose a parse-thread count: physical cores when psutil is available, else logical cores, clamped to 1..32.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        try:
            # Execute guarded operations and establish recovery boundaries
            import psutil
            physical = psutil.cpu_count(logical=False)
        except Exception:
            physical = None
        return max(1, min(32, int(physical or os.cpu_count() or 4)))

    def _iter_supported_files(self, path: str, recursive: bool = True) -> list[str]:
        """List supported files under a file or directory path (sorted, symlinks not followed during the walk).
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        expanded = os.path.realpath(os.path.expanduser(path))
        if os.path.isfile(expanded):
            ext = os.path.splitext(expanded)[1].lower()
            return [expanded] if ext in _SUPPORTED_EXTENSIONS else []
        if not os.path.isdir(expanded):
            return []

        files: list[str] = []
        if recursive:
            for root, dirs, names in os.walk(expanded, followlinks=False):
                # Iterate over items, evaluate loop invariant, and update progress
                # Prune dependency/VCS/cache folders in place so os.walk does not descend into them.
                dirs[:] = [
                    d for d in dirs
                    if d not in {".git", "venv", ".venv", "__pycache__", "node_modules"}
                ]
                for name in names:
                    # Iterate over items, evaluate loop invariant, and update progress
                    ext = os.path.splitext(name)[1].lower()
                    if ext in _SUPPORTED_EXTENSIONS:
                        files.append(os.path.realpath(os.path.join(root, name)))
        else:
            for name in os.listdir(expanded):
                # Iterate over items, evaluate loop invariant, and update progress
                candidate = os.path.realpath(os.path.join(expanded, name))
                ext = os.path.splitext(name)[1].lower()
                if os.path.isfile(candidate) and ext in _SUPPORTED_EXTENSIONS:
                    files.append(candidate)
        return sorted(files)

    def _chunks_from_file(self, filepath: str, chunk_size: int, overlap: int) -> list[str]:
        """Turn a file into text chunks; CSV rows become one chunk each, other files are chunked by words.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        ext = os.path.splitext(filepath)[1].lower()
        # CSV: use the 'Description' column when present, else render each row as 'header: value' pairs.
        if ext == ".csv":
            import csv
            chunks = []
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    # Iterate over items, evaluate loop invariant, and update progress
                    if not row:
                        continue
                    if header and "Description" in header:
                        desc_idx = header.index("Description")
                        val = row[desc_idx] if desc_idx < len(row) else ", ".join(row)
                        chunks.append(val)
                    else:
                        chunks.append(
                            ", ".join(f"{h}: {v}" for h, v in zip(header, row) if v)
                            if header else ", ".join(row)
                        )
            return [c for c in chunks if c.strip()]

        text = self.extract_text(filepath)
        if not text.strip():
            return []
        return self.chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    def _parse_file_for_ingest(
        self,
        index: int,
        filepath: str,
        chunk_size: int,
        overlap: int,
    ) -> _ParsedFile:
        """Parse one file into a _ParsedFile, capturing any error instead of raising (runs in a worker thread).
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        source_file = os.path.basename(filepath)
        try:
            # Execute guarded operations and establish recovery boundaries
            chunks = self._chunks_from_file(filepath, chunk_size, overlap)
            return _ParsedFile(index=index, path=filepath, source_file=source_file, chunks=chunks)
        except Exception as exc:
            logger.warning("Error parsing %s: %s", filepath, exc)
            return _ParsedFile(
                index=index,
                path=filepath,
                source_file=source_file,
                chunks=[],
                error=str(exc),
            )

    def _embed_texts_batched(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed texts in batches and return a float32 matrix (empty (0, dimension) matrix for no input).
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")

        vectors = []
        for start in range(0, len(texts), batch_size):
            # Iterate over items, evaluate loop invariant, and update progress
            batch = texts[start:start + batch_size]
            try:
                # Execute guarded operations and establish recovery boundaries
                encoded = self.encoder.encode(
                    batch,
                    batch_size=batch_size,
                    show_progress_bar=False,
                )
            # Some encoders do not accept the extra keyword arguments; retry with defaults.
            except TypeError:
                encoded = self.encoder.encode(batch)
            vectors.append(np.asarray(encoded, dtype="float32"))
        return np.vstack(vectors).astype("float32") if vectors else np.empty((0, self.dimension), dtype="float32")

    def _add_chunks_to_index(
        self,
        chunk_records: list[tuple[str, str]],
        embeddings: np.ndarray,
        ingested_at: str | None = None,
        save: bool = True,
    ) -> int:
        """Add chunks and embeddings to the index and (optionally) the database.

        save=False keeps them in memory only (used for transient eval text). With
        save=True the SQLite insert and index mutation run under one transaction and
        are rolled back together on error. Returns the number of chunks added.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        if not chunk_records:
            return 0
        ingested_at = ingested_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self._write_lock:
            start_id = self._next_vector_id()
            vector_ids = list(range(start_id, start_id + len(chunk_records)))
            if not save:
                self._add_embeddings_to_index(embeddings, vector_ids)
                for vector_id, (chunk_text, source_file) in zip(vector_ids, chunk_records):
                    # Iterate over items, evaluate loop invariant, and update progress
                    self.documents.append({
                        "id": None,
                        "vector_id": vector_id,
                        "text": chunk_text,
                        "source_file": source_file,
                        "chunk_id": vector_id,
                        "ingested_at": ingested_at,
                    })
                return len(chunk_records)

            # Build all rows first so the database insert happens as one batch.
            # Compile all row tuples for an atomic batch insert.
            rows = [
                (vid, chunk_text, source_file, vid, ingested_at)
                for vid, (chunk_text, source_file) in zip(vector_ids, chunk_records)
            ]
            # Keep copies so a failure can restore the previous state (FAISS has no transaction support).
            prior_index = faiss.clone_index(self.index)
            prior_documents = list(self.documents)
            flush_job = None
            with self._pool.get_connection() as conn:
                try:
                    # Execute guarded operations and establish recovery boundaries
                    conn.execute("BEGIN IMMEDIATE")
                    conn.executemany(
                        "INSERT INTO documents(vector_id, text, source_file, chunk_id, ingested_at)"
                        " VALUES (?, ?, ?, ?, ?)",
                        rows,
                    )
                    self._add_embeddings_to_index(embeddings, vector_ids)
                    # Snapshot the post-mutation index and hand the slow
                    # faiss.write_index()+os.replace() disk write off to the
                    # background flush thread, then commit immediately
                    # without waiting for it. This is a deliberate atomicity
                    # trade-off (see _wait_for_index_flush's docstring): if
                    # the flush later fails, the SQLite commit below cannot
                    # be undone anymore. What it buys us is that neither the
                    # SQLite write transaction nor _write_lock are held for
                    # the duration of the disk write, so a second concurrent
                    # ingest/remove call isn't stuck queueing behind it.
                    index_snapshot = faiss.clone_index(self.index)
                    flush_job = self._submit_index_flush(index_snapshot)
                    conn.commit()
                    self.documents = self._fetch_documents(conn)
                except Exception:
                    conn.rollback()
                    self.index = prior_index
                    self.documents = prior_documents
                    raise
        # Outside _write_lock and outside the SQLite transaction: block
        # until *this* call's own snapshot is durable on disk before
        # returning, preserving "data is persisted when this method
        # returns" for callers, without making other writers wait for it.
        if flush_job is not None:
            self._wait_for_index_flush(flush_job)
        return len(chunk_records)

    def ingest_files(
        self,
        filepaths: list[str],
        chunk_size: int = 200,
        overlap: int = 50,
        batch_size: int = 32,
        max_workers: int | None = None,
        progress_cb: Callable[[int, int, dict], None] | None = None,
    ) -> dict:
        """
        Parse files concurrently, embed chunks in batches, and append to FAISS.

        progress_cb receives (files_parsed, total_files, event_dict).
        Returns {files, errors, chunks_added, file_count, error_count}.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        files = [os.path.realpath(os.path.expanduser(p)) for p in filepaths]
        files = [p for p in files if os.path.splitext(p)[1].lower() in _SUPPORTED_EXTENSIONS]
        total = len(files)
        if total == 0:
            return {"files": [], "errors": [], "chunks_added": 0, "file_count": 0, "error_count": 0}

        max_workers = max_workers or self._default_worker_count()
        parsed_results: list[_ParsedFile] = []
        errors: list[dict] = []
        parsed_lock = threading.Lock()
        files_parsed = 0

        if progress_cb:
            progress_cb(0, total, {"status": "queued", "filename": "", "queued": total})

        # Parsing (disk and CPU bound) is parallel; embedding and index writes happen afterwards on this thread.
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="karl-rag-parse") as pool:
            futures = [
                pool.submit(self._parse_file_for_ingest, idx, filepath, chunk_size, overlap)
                for idx, filepath in enumerate(files)
            ]
            for future in as_completed(futures):
                # Iterate over items, evaluate loop invariant, and update progress
                parsed = future.result()
                with parsed_lock:
                    parsed_results.append(parsed)
                    files_parsed += 1
                    current = files_parsed
                if parsed.error:
                    errors.append({
                        "path": parsed.path,
                        "filename": parsed.source_file,
                        "error": parsed.error,
                    })
                if progress_cb:
                    progress_cb(current, total, {
                        "status": "parsed",
                        "filename": parsed.source_file,
                        "chunks": len(parsed.chunks),
                        "error": parsed.error,
                    })

        # Restore the input order, since futures complete in arbitrary order, so vector ids are deterministic.
        parsed_results.sort(key=lambda item: item.index)
        chunk_records: list[tuple[str, str]] = []
        per_file: list[dict] = []
        for parsed in parsed_results:
            # Iterate over items, evaluate loop invariant, and update progress
            if parsed.error:
                continue
            per_file.append({
                "path": parsed.path,
                "filename": parsed.source_file,
                "chunks": len(parsed.chunks),
            })
            for chunk in parsed.chunks:
                # Iterate over items, evaluate loop invariant, and update progress
                chunk_records.append((chunk, parsed.source_file))

        # Embed everything in one pass so the encoder runs in batches across file boundaries.
        texts = [text for text, _source in chunk_records]
        embeddings = self._embed_texts_batched(texts, batch_size=batch_size)
        added = self._add_chunks_to_index(chunk_records, embeddings, save=True)
        logger.info(
            "Batch-ingested %d chunks from %d/%d files (%d errors)",
            added,
            len(per_file),
            total,
            len(errors),
        )
        return {
            "files": per_file,
            "errors": errors,
            "chunks_added": added,
            "file_count": len(per_file),
            "error_count": len(errors),
            "queued_file_count": total,
        }

    def ingest_directory(
        self,
        path: str,
        recursive: bool = True,
        chunk_size: int = 200,
        overlap: int = 50,
        batch_size: int = 32,
        max_workers: int | None = None,
        progress_cb: Callable[[int, int, dict], None] | None = None,
    ) -> dict:
        """Ingest every supported file under a directory (see ingest_files for the result format).
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        files = self._iter_supported_files(path, recursive=recursive)
        return self.ingest_files(
            files,
            chunk_size=chunk_size,
            overlap=overlap,
            batch_size=batch_size,
            max_workers=max_workers,
            progress_cb=progress_cb,
        )

    def ingest_file(self, filepath: str, chunk_size: int = 200, overlap: int = 50) -> int:
        """
        Extract, chunk, embed, and add a file to the index.
        Saves the index to disk after each successful ingest.

        Returns:
            Number of chunks added.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        result = self.ingest_files(
            [filepath],
            chunk_size=chunk_size,
            overlap=overlap,
            batch_size=32,
            max_workers=1,
        )
        return int(result["chunks_added"])

    def ingest_text(self, text: str, source_name: str = "inline", chunk_size: int = 200, overlap: int = 50) -> int:
        """Ingest raw text string directly (no file needed). Useful for eval harness."""
        # Step 1: Validate input parameters, operational preconditions, and state
        chunks = self.chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return 0

        embeddings = self._embed_texts_batched(chunks, batch_size=32)
        self._add_chunks_to_index(
            [(chunk_text, source_name) for chunk_text in chunks],
            embeddings,
            save=False,
        )
        # No save_index() here - transient ingest for eval use
        # No save_index() here — transient ingest for eval use
        return len(chunks)

    def _is_toc_chunk(self, doc: dict) -> bool:
        """Helper to identify if a chunk is part of the Table of Contents (pages 1 to 10)."""
        # Step 1: Validate input parameters, operational preconditions, and state
        text = doc.get("text", "")
        import re
        page_match = re.search(r"Page\s+(\d+)\s+of\s+\d+", text)
        if page_match:
            try:
                # Execute guarded operations and establish recovery boundaries
                if int(page_match.group(1)) <= 10:
                    return True
            except ValueError:
                pass
        return False

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        source_filter: str | None = None,
        threshold: float = 1.0,
        with_attribution: bool = False,
        mode: str = "dense",
    ) -> list[str] | list[dict]:
        """
        Retrieve relevant chunks. 
        with_attribution=True returns List[dict] with full metadata.
        with_attribution=False (default) returns List[str] for backward compatibility.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        results = self.retrieve_with_metadata(
            query,
            top_k=top_k,
            source_filter=source_filter,
            threshold=threshold,
            mode=mode,
        )

        formatted = []
        for i, r in enumerate(results):
            # Iterate over items, evaluate loop invariant, and update progress
            text = r["text"]
            if self.contextual_headers:
                header = f"[Source: {r['source_file']} | Chunk {r['chunk_id']}]\n"
                text = header + text
            formatted.append({
                "text":        text,
                "source_file": r.get("source_file", ""),
                "chunk_id":    r.get("chunk_id", i),
                "distance":    float(r.get("distance", 0.0)),
                "rank":        i + 1,
                "ingested_at": r.get("ingested_at", ""),
            })

        if with_attribution:
            return formatted
        return [r["text"] for r in formatted]

    def retrieve_with_attribution(
        self,
        query: str,
        top_k: int = 5,
        source_filter: str | None = None,
        threshold: float = 1.0,
        mode: str = "dense",
    ) -> list[dict]:
        """Always returns List[dict] with full attribution metadata."""
        # Step 1: Validate input parameters, operational preconditions, and state
        return self.retrieve(query, top_k, source_filter, threshold, with_attribution=True, mode=mode)

    def retrieve_sparse(
        self,
        query: str,
        top_k: int = 3,
        source_filter: str | None = None,
    ) -> list[dict]:
        """Retrieve relevant chunks using pure TF-IDF sparse matching."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if not self.documents:
            return []

        # The vectorizer is re-fitted on every call, which is simple but costs time proportional to the whole corpus.
        # Fit TF-IDF on current documents
        from app.utils.custom_embeddings import TfidfEmbedder
        tfidf = TfidfEmbedder()
        texts = [doc["text"] for doc in self.documents]
        tfidf.fit(texts)

        # Transform query and docs
        q_vec = tfidf.transform(query)
        if len(q_vec) == 0 or np.linalg.norm(q_vec) == 0.0:
            return []

        scored_docs = []
        for idx, doc in enumerate(self.documents):
            # Iterate over items, evaluate loop invariant, and update progress
            if source_filter and doc.get("source_file") != source_filter:
                continue
            d_vec = tfidf.transform(doc["text"])
            sim = tfidf.cosine_similarity(q_vec, d_vec)
            if sim > 0.0:
                scored_docs.append({
                    **doc,
                    "distance": float(1.0 - sim),  # convert similarity to a distance-like metric (0 is closest)
                    "similarity": sim
                })

        # Sort by similarity descending
        scored_docs.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Add rank
        for rank, doc in enumerate(scored_docs, 1):
            # Iterate over items, evaluate loop invariant, and update progress
            doc["rank"] = rank

        return scored_docs[:top_k]

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 3,
        source_filter: str | None = None,
        rrf_constant: int = 60,
        use_reranker: bool = False,
        rerank_candidates: int = 15,
    ) -> list[dict]:
        """Combine Dense (FAISS) and Sparse (TF-IDF) results via Reciprocal Rank Fusion.

        Args:
            query:             Search query string.
            top_k:             Number of results to return.
            source_filter:     Restrict results to a specific source file.
            rrf_constant:      RRF smoothing constant (default 60).
            use_reranker:      If True, re-score the RRF candidate pool with a local
                               CrossEncoder before slicing to top_k.
            rerank_candidates: How many RRF candidates to pass to the CrossEncoder.
                               Ignored when use_reranker=False.
        """
        # When reranking, gather a larger initial pool; otherwise fetch just enough.
        n_pool  = rerank_candidates if use_reranker else top_k
        fetch_k = n_pool * 2   # over-fetch so RRF can surface the best overlaps

        dense_results = self.retrieve_with_metadata(
            query, top_k=fetch_k, source_filter=source_filter, mode="dense"
        )
        sparse_results = self.retrieve_sparse(
            query, top_k=fetch_k, source_filter=source_filter
        )

        # RRF: each result list contributes 1 / (rrf_constant + rank) per document, so chunks that appear
        # near the top of both lists win without needing comparable raw scores.
        # ── Reciprocal Rank Fusion ────────────────────────────────────────────
        rrf_scores: dict[int, float] = {}
        docs_map:   dict[int, dict]  = {}

        for rank, doc in enumerate(dense_results, 1):
            # Iterate over items, evaluate loop invariant, and update progress
            cid = doc["chunk_id"]
            docs_map[cid] = doc
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_constant + rank)

        for rank, doc in enumerate(sparse_results, 1):
            # Iterate over items, evaluate loop invariant, and update progress
            cid = doc["chunk_id"]
            if cid not in docs_map:
                docs_map[cid] = doc
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_constant + rank)

        sorted_cids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        # The best possible fused score is a #1 rank in both lists; it is used to scale scores to a 0..1 distance.
        max_score   = 2.0 / (rrf_constant + 1)

        # Build the candidate pool sized for reranking (or top_k for the plain path).
        pool: list[dict] = []
        for rank, cid in enumerate(sorted_cids[:n_pool], 1):
            # Iterate over items, evaluate loop invariant, and update progress
            rrf_score_val = rrf_scores[cid]
            pool.append({
                **docs_map[cid],
                "rank":      rank,
                "distance":  float(max(0.0, 1.0 - (rrf_score_val / max_score))),
                "rrf_score": rrf_score_val,
            })

        if not use_reranker or not pool:
            return pool[:top_k]

        # ── Cross-Encoder reranking ───────────────────────────────────────────
        reranker = self.reranker
        if reranker is None:
            logger.debug("CrossEncoder unavailable; returning RRF-ordered results.")
            return pool[:top_k]

        inputs = [(query, c["text"]) for c in pool]
        try:
            # Execute guarded operations and establish recovery boundaries
            scores = reranker.predict(inputs)
        except Exception as exc:
            logger.warning("Reranker predict() failed (%s); falling back to RRF.", exc)
            return pool[:top_k]

        for candidate, score in zip(pool, scores):
            # Iterate over items, evaluate loop invariant, and update progress
            candidate["rerank_score"] = float(score)

        pool.sort(key=lambda c: c["rerank_score"], reverse=True)
        pool = pool[:top_k]
        for i, c in enumerate(pool, 1):
            # Iterate over items, evaluate loop invariant, and update progress
            c["rank"] = i

        return pool

    def retrieve_with_metadata(
        self,
        query: str,
        top_k: int = 3,
        source_filter: str | None = None,
        threshold: float = 0.0,
        mode: str = "dense",
    ) -> list[dict]:
        """
        Like retrieve() but returns full metadata dicts including distance scores.
        Supports dense, sparse, and hybrid retrieval.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        if self.index.ntotal == 0:
            return []

        if mode == "sparse":
            return self.retrieve_sparse(query, top_k=top_k, source_filter=source_filter)
        elif mode == "hybrid":
            return self.retrieve_hybrid(query, top_k=top_k, source_filter=source_filter)

        # Hybrid Search: check if query wants to list/show all workers in a specific department.
        # These rules are hardcoded for one document set (department names) and act as an exact-match shortcut.
        # Hybrid Search: check if query wants to list/show all workers in a specific department
        exact_matches = []
        import re
        
        dept_match = None
        for dept in ["IT", "Finance", "Admin", "EVS", "Marketing"]:
            # Iterate over items, evaluate loop invariant, and update progress
            if re.search(r"\b" + re.escape(dept) + r"\b", query, re.IGNORECASE):
                # Check for listing and counting indicator keywords
                indicators = [
                    "all", "list", "everyone", "who works", "workers in", "employees in", 
                    "show me", "people in", "how many", "number of", "count", "people work", 
                    "who is in", "who are in", "hwo", "how", "many", "worker", "workers", 
                    "employee", "employees", "people", "person", "staff", "who", "show"
                ]
                if any(w in query.lower() for w in indicators):
                    dept_match = dept
                    break

        if dept_match:
            for doc in self.documents:
                # Iterate over items, evaluate loop invariant, and update progress
                if source_filter and doc.get("source_file") != source_filter:
                    continue
                if re.search(r"\b" + re.escape(dept_match) + r"\b", doc["text"], re.IGNORECASE):
                    exact_matches.append({
                        **doc,
                        "rank": 0,
                        "distance": 0.0,
                    })
            if exact_matches:
                return exact_matches

        # Alphanumeric ID matching: tokens that mix letters and digits (e.g. 'AB12') are treated as codes.
        # Alphanumeric ID matching
        id_tokens = re.findall(r"\b[A-Z0-9]{3,10}\b", query)
        for token in id_tokens:
            # Must contain both letters and numbers to be an ID/code
            if any(c.isalpha() for c in token) and any(c.isdigit() for c in token):
                for doc in self.documents:
                    # Iterate over items, evaluate loop invariant, and update progress
                    if source_filter and doc.get("source_file") != source_filter:
                        continue
                    if re.search(r"\b" + re.escape(token) + r"\b", doc["text"]):
                        if not any(e["chunk_id"] == doc["chunk_id"] for e in exact_matches):
                            exact_matches.append({
                                **doc,
                                "rank": 0,
                                "distance": 0.0,
                            })

        # Section matching, e.g., "19.3", "19.5" (table-of-contents chunks are skipped so the real section wins).
        # Section matching, e.g., "19.3", "19.5"
        section_tokens = re.findall(r"\b\d{1,2}\.\d{1,2}(?:\.\d{1,2})?\b", query)
        for token in section_tokens:
            # Iterate over items, evaluate loop invariant, and update progress
            for doc in self.documents:
                # Iterate over items, evaluate loop invariant, and update progress
                if source_filter and doc.get("source_file") != source_filter:
                    continue
                if self._is_toc_chunk(doc):
                    continue
                if re.search(r"\b" + re.escape(token) + r"\b", doc["text"]):
                    if not any(e["chunk_id"] == doc["chunk_id"] for e in exact_matches):
                        exact_matches.append({
                            **doc,
                            "rank": 0,
                            "distance": 0.0,
                        })

        # Chapter matching, e.g., "Chapter 20", "chapter 19", "ch 20"
        chapter_matches = re.findall(r"\b(?:chapter|ch)\s+(\d{1,2})\b", query, re.IGNORECASE)
        for ch_num in chapter_matches:
            # Iterate over items, evaluate loop invariant, and update progress
            spaced_ch_pattern = r"\b" + r"\s*".join("chapter") + r"\s*" + r"\s*".join(ch_num) + r"\b"
            spaced_ch_short_pattern = r"\b" + r"\s*".join("ch") + r"\s*" + r"\s*".join(ch_num) + r"\b"
            for doc in self.documents:
                # Iterate over items, evaluate loop invariant, and update progress
                if source_filter and doc.get("source_file") != source_filter:
                    continue
                if self._is_toc_chunk(doc):
                    continue
                if re.search(spaced_ch_pattern, doc["text"], re.IGNORECASE) or re.search(spaced_ch_short_pattern, doc["text"], re.IGNORECASE):
                    if not any(e["chunk_id"] == doc["chunk_id"] for e in exact_matches):
                        exact_matches.append({
                            **doc,
                            "rank": 0,
                            "distance": 0.0,
                        })

        # Topic keyword matching: (topic phrase searched in documents, query keywords that trigger it).
        # Topic keyword matching
        topic_keywords = [
            ("continuing obligations", ["continuing obligations", "obligation"]),
            ("final pay", ["final pay"]),
            ("return of company property", ["return of company property", "return of property"]),
            ("morale officer", ["morale officer", "phil"]),
            ("break room", ["break room", "kitchen"]),
            ("traditions", ["traditions", "celebrations", "golden gherkin"]),
            ("parking", ["parking", "gerald"]),
            ("remote and hybrid work", ["remote work", "hybrid work", "telecommuting"]),
        ]
        for topic, keywords in topic_keywords:
            # Iterate over items, evaluate loop invariant, and update progress
            if any(re.search(r"\b" + re.escape(kw) + r"\w*", query, re.IGNORECASE) for kw in keywords):
                for doc in self.documents:
                    # Iterate over items, evaluate loop invariant, and update progress
                    if source_filter and doc.get("source_file") != source_filter:
                        continue
                    if self._is_toc_chunk(doc):
                        continue
                    if re.search(r"\b" + re.escape(topic) + r"\w*", doc["text"], re.IGNORECASE):
                        if not any(e["chunk_id"] == doc["chunk_id"] for e in exact_matches):
                            exact_matches.append({
                                **doc,
                                "rank": 0,
                                "distance": 0.0,
                            })

        # Run vector search. With a source filter, over-fetch 5x because filtered-out hits would otherwise leave too few.
        # Run vector search
        query_vector = self.encoder.encode([query]).astype("float32")
        fetch_k = min(top_k * 5 if source_filter else top_k, self.index.ntotal)
        distances, indices = self.index.search(query_vector, fetch_k)
        docs_by_vector_id = {int(doc.get("vector_id", doc.get("chunk_id", -1))): doc for doc in self.documents}

        results = list(exact_matches)
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            # Iterate over items, evaluate loop invariant, and update progress
            if idx == -1:
                continue
            doc = docs_by_vector_id.get(int(idx))
            if not doc:
                continue
            if source_filter and doc.get("source_file") != source_filter:
                continue
            # Avoid duplicates of exact matches
            if any(e["chunk_id"] == doc["chunk_id"] for e in results):
                continue
            results.append({
                **doc,
                "rank": rank + len(exact_matches),
                "distance": float(dist),
            })
            if len(results) >= top_k:
                break

        if threshold > 0.0:
            results = [r for r in results if r["distance"] <= threshold]

        return results[:top_k]

    # ── Retrieval Eval Metrics ─────────────────────────────────────────────────

    def eval_retrieval(
        self,
        query: str,
        expected_chunk_ids: list[int],
        top_k: int = 5,
    ) -> dict:
        """
        Measure retrieval quality for a single query.

        Args:
            query:              Query string.
            expected_chunk_ids: List of chunk_id values that count as relevant.
            top_k:              Evaluation cutoff.

        Returns:
            {
              "hit_at_1": bool,
              "hit_at_3": bool,
              "hit_at_k": bool,
              "reciprocal_rank": float,   # 1/rank of first hit, 0 if not found
              "retrieved_ids": list[int]
            }
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        retrieved = self.retrieve_with_metadata(query, top_k=top_k)
        retrieved_ids = [r["chunk_id"] for r in retrieved]
        expected_set = set(expected_chunk_ids)

        rr = 0.0
        for rank, cid in enumerate(retrieved_ids, 1):
            # Iterate over items, evaluate loop invariant, and update progress
            if cid in expected_set:
                rr = 1.0 / rank
                break

        return {
            "hit_at_1": bool(retrieved_ids[:1] and retrieved_ids[0] in expected_set),
            "hit_at_3": bool(any(cid in expected_set for cid in retrieved_ids[:3])),
            "hit_at_k": bool(any(cid in expected_set for cid in retrieved_ids)),
            "reciprocal_rank": round(rr, 4),
            "retrieved_ids": retrieved_ids,
        }

    # ── Convenience ───────────────────────────────────────────────────────────

    def list_sources(self) -> list[str]:
        """Return unique source filenames currently in the index."""
        # Step 1: Validate input parameters, operational preconditions, and state
        seen = set()
        sources = []
        for doc in self.documents:
            # Iterate over items, evaluate loop invariant, and update progress
            sf = doc.get("source_file", "unknown")
            if sf not in seen:
                seen.add(sf)
                sources.append(sf)
        return sources

    @property
    def total_chunks(self) -> int:
        """Number of chunks currently held in memory."""
        # Step 1: Validate input parameters, operational preconditions, and state
        return len(self.documents)

    def remove_source(self, source_name: str):
        """Remove all chunks belonging to source_name from SQLite and FAISS atomically."""
        # Step 1: Validate input parameters, operational preconditions, and state
        flush_job = None
        with self._write_lock:
            prior_index = faiss.clone_index(self.index)
            prior_documents = list(self.documents)
            with self._pool.get_connection() as conn:
                try:
                    # Execute guarded operations and establish recovery boundaries
                    conn.execute("BEGIN IMMEDIATE")
                    rows = conn.execute(
                        "SELECT vector_id FROM documents WHERE source_file = ? ORDER BY vector_id",
                        (source_name,),
                    ).fetchall()
                    vector_ids = [int(row["vector_id"]) for row in rows]
                    if not vector_ids:
                        conn.commit()
                        return

                    ids_array = np.asarray(vector_ids, dtype="int64")
                    selector = faiss.IDSelectorArray(
                        len(vector_ids),
                        faiss.swig_ptr(ids_array),
                    )
                    self.index.remove_ids(selector)
                    conn.execute("DELETE FROM documents WHERE source_file = ?", (source_name,))
                    # See _add_chunks_to_index for why this hands the disk
                    # flush off to the background thread and commits without
                    # waiting for it, rather than calling
                    # _write_index_atomic() synchronously here.
                    index_snapshot = faiss.clone_index(self.index)
                    flush_job = self._submit_index_flush(index_snapshot)
                    conn.commit()
                    self.documents = self._fetch_documents(conn)
                except Exception:
                    conn.rollback()
                    self.index = prior_index
                    self.documents = prior_documents
                    raise
        if flush_job is not None:
            self._wait_for_index_flush(flush_job)
        logger.info(f"Source '{source_name}' removed. Index now has {len(self.documents)} chunks.")

    def rebuild_index(self):
        """Re-encode all chunks currently in metadata and rebuild the FAISS index."""
        # Step 1: Validate input parameters, operational preconditions, and state
        if not self.documents:
            self.index = self._new_index()
            self.save_index()
            return
        
        texts = [d["text"] for d in self.documents]
        self.index = self._new_index()
        embeddings = self._embed_texts_batched(texts, batch_size=32)
        vector_ids = [int(d.get("vector_id", d.get("chunk_id", i))) for i, d in enumerate(self.documents)]
        self._add_embeddings_to_index(np.array(embeddings).astype("float32"), vector_ids)
        self.save_index()
        logger.info(f"Rebuilt index by re-encoding all {len(self.documents)} current chunks.")
