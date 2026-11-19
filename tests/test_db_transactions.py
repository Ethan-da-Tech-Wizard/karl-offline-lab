# ==============================================================================
# Program: test_db_transactions.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Comprehensive test suite exercising db transactions functionality.
#          Verifies state transitions, error boundaries, regression resistance,
#          and contract compliance under tests.
# ==============================================================================

import os
import sqlite3
import threading
import time

import numpy as np
import pytest

from app.utils.rag_pipeline import RAGPipeline


class _DeterministicEncoder:
    """Hash-based bag-of-words embedder (384 dims, L2-normalised) replacing the real sentence encoder, so the tests need no model download and give repeatable vectors."""
    def encode(self, texts, batch_size=32, show_progress_bar=False):
        """Embed one or more texts by hashing each token into a fixed 384-dim vector, then normalise."""
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for text in texts:
            vec = np.zeros(384, dtype="float32")
            for token in text.lower().split():
                vec[hash(token) % 384] += 1.0
            norm = np.linalg.norm(vec)
            if norm:
                vec = vec / norm
            vectors.append(vec)
        return np.vstack(vectors)


class _TestRAG(RAGPipeline):
    """RAGPipeline whose encoder is replaced by the deterministic one."""
    def __init__(self, *args, **kwargs):
        """Construct the pipeline normally, then inject the fake encoder."""
        super().__init__(*args, **kwargs)
        self._encoder = _DeterministicEncoder()


class _FailAfterFaissAddRAG(_TestRAG):
    """Pipeline that adds vectors to FAISS and then raises, to prove ingest rolls back both stores."""
    def _add_embeddings_to_index(self, embeddings, vector_ids):
        """Do the real FAISS add, then fail on purpose (the worst moment: SQLite rows exist and FAISS already changed)."""
        super()._add_embeddings_to_index(embeddings, vector_ids)
        raise RuntimeError("forced FAISS add failure")


class _BlockingAddRAG(_TestRAG):
    """Pipeline whose FAISS add can be paused mid-ingest (via events) to hold a transaction open while another reader runs."""
    def __init__(self, *args, entered_add: threading.Event, release_add: threading.Event, **kwargs):
        """Store the two threading events used to signal 'entered add' and 'allowed to continue'."""
        super().__init__(*args, **kwargs)
        self.entered_add = entered_add
        self.release_add = release_add
        self.block_add = False

    def _add_embeddings_to_index(self, embeddings, vector_ids):
        """When blocking is on: announce entry, wait (max 5s) for release, then perform the real add."""
        if not self.block_add:
            return super()._add_embeddings_to_index(embeddings, vector_ids)
        self.entered_add.set()
        if not self.release_add.wait(timeout=5.0):
            raise RuntimeError("timed out waiting to release FAISS add")
        return super()._add_embeddings_to_index(embeddings, vector_ids)


def _write_doc(path, name, body):
    """Write a text file under path and return its full path."""
    doc_path = os.path.join(path, name)
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(body)
    return doc_path


def _db_count(path):
    """Count rows in the documents table via a separate SQLite connection, i.e. what an independent reader would see."""
    conn = sqlite3.connect(os.path.join(path, "meta.db"), timeout=1.0)
    try:
        return conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    finally:
        conn.close()


def test_ingest_rolls_back_sqlite_and_faiss_when_faiss_add_fails(tmp_path):
    """Atomicity: if adding vectors to FAISS fails after the SQLite insert, the whole ingest rolls back; the FAISS index size, in-memory documents and the SQLite table must all be unchanged."""
    pipeline = _FailAfterFaissAddRAG(index_path=str(tmp_path))
    doc_path = _write_doc(
        str(tmp_path),
        "rollback.txt",
        "alpha beta gamma delta epsilon zeta eta theta iota kappa",
    )

    # Capture the FAISS size first: the failure is injected AFTER the real add, so only a true rollback can bring ntotal back to this value.
    before_count = pipeline.index.ntotal
    with pytest.raises(RuntimeError, match="forced FAISS add failure"):
        # A 4-word window with 1 word overlap over ten words yields several chunks, so the failed add covers multiple vectors rather than a trivial single insert.
        pipeline.ingest_file(doc_path, chunk_size=4, overlap=1)

    assert pipeline.index.ntotal == before_count
    # The in-memory document list is a third copy of the data; it must be rolled back too or later retrievals would index into rows that no longer exist.
    assert pipeline.documents == []
    assert _db_count(str(tmp_path)) == 0


def test_wal_allows_metadata_reads_while_ingest_transaction_is_open(tmp_path):
    """SQLite WAL mode: while one thread holds an ingest transaction open, another pipeline can still read committed metadata and run retrieval quickly (<1s) without 'database is locked'; after release the writer's row appears too."""
    seed_path = _write_doc(
        str(tmp_path),
        "seed.txt",
        "stable searchable metadata chunk about neural vector search",
    )
    writer_path = _write_doc(
        str(tmp_path),
        "writer.txt",
        "background writer chunk that should not block existing reads",
    )

    entered_add = threading.Event()
    release_add = threading.Event()
    pipeline = _BlockingAddRAG(
        index_path=str(tmp_path),
        entered_add=entered_add,
        release_add=release_add,
    )
    pipeline.ingest_file(seed_path, chunk_size=20, overlap=0)
    assert _db_count(str(tmp_path)) == 1
    # Only block after the seed ingest: the seed must complete normally so the reader has committed data to find while the writer is stalled.
    pipeline.block_add = True

    errors = []

    def ingest_in_background():
        """Run the blocked ingest on a thread and collect any exception it raises."""
        try:
            pipeline.ingest_file(writer_path, chunk_size=20, overlap=0)
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=ingest_in_background)
    worker.start()
    # Proves the writer is genuinely mid-transaction (rows inserted, FAISS add pending) before the reader starts; otherwise the test would pass vacuously.
    assert entered_add.wait(timeout=5.0)

    start = time.perf_counter()
    reader = _TestRAG(index_path=str(tmp_path))
    # WAL readers see the last committed snapshot, so exactly the seed row (1) is visible and the uncommitted writer row is not.
    rows_seen = _db_count(str(tmp_path))
    results = reader.retrieve_with_metadata("neural vector search", top_k=1, threshold=2.0)
    elapsed = time.perf_counter() - start

    release_add.set()
    worker.join(timeout=5.0)

    assert not errors
    assert not worker.is_alive()
    assert rows_seen == 1
    # A rollback-journal database would raise 'database is locked' or wait out the busy timeout here; finishing in under a second shows the reader never blocked.
    assert elapsed < 1.0
    assert results
    assert results[0]["source_file"] == "seed.txt"
    assert _db_count(str(tmp_path)) == 2


def test_remove_source_deletes_matching_sqlite_rows_and_faiss_vectors(tmp_path):
    """remove_source deletes the source's rows from SQLite and its vectors from FAISS while leaving other sources intact and list_sources consistent."""
    pipeline = _TestRAG(index_path=str(tmp_path))
    keep_path = _write_doc(str(tmp_path), "keep.txt", "keep neural vector search chunk")
    drop_path = _write_doc(str(tmp_path), "drop.txt", "drop processor instruction chunk")

    pipeline.ingest_file(keep_path, chunk_size=20, overlap=0)
    pipeline.ingest_file(drop_path, chunk_size=20, overlap=0)
    assert pipeline.index.ntotal == 2
    # After release the writer commits, confirming it was merely paused, not aborted by the concurrent reader.
    assert _db_count(str(tmp_path)) == 2

    # Two sources with distinct vocabulary are ingested so that deleting one cannot be confused with deleting all; FAISS is expected to drop exactly the vectors owned by that source.
    pipeline.remove_source("drop.txt")

    conn = sqlite3.connect(os.path.join(str(tmp_path), "meta.db"))
    try:
        sources = [
            row[0]
            for row in conn.execute("SELECT source_file FROM documents ORDER BY source_file")
        ]
    finally:
        conn.close()

    assert pipeline.index.ntotal == 1
    assert sources == ["keep.txt"]
    assert pipeline.list_sources() == ["keep.txt"]
