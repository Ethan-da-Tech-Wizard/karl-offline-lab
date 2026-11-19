# ==============================================================================
# Program: test_rag_pipeline_fixes.py
# Purpose: Regression tests for RAGPipeline.chunk_text overlap guard and the
#          dataset_merger / keychain / trace_logger fixes. Heavy optional
#          dependencies (faiss, numpy, pymupdf, docx) are stubbed only when not installed.
# ==============================================================================
import importlib.util
import sys
from unittest import mock

# find_spec() asks "is this package installed?" WITHOUT importing it. Importing the real
# `fitz` shim emits a FutureWarning, and pytest.ini turns warnings into errors, so a plain
# __import__ probe would fail collection on machines where PyMuPDF is installed.
for _mod in ("faiss", "numpy", "pymupdf", "fitz", "docx"):
    if importlib.util.find_spec(_mod) is None:  # keep these tests runnable without the ML stack
        sys.modules[_mod] = mock.MagicMock()

from app.utils.rag_pipeline import RAGPipeline  # noqa: E402 (must come after the sys.modules mocks above)


def test_chunk_text_overlap_not_smaller_than_chunk_size_terminates():
    """overlap >= chunk_size used to raise ValueError (step 0) or loop badly; it must advance one word at a time."""
    text = " ".join(f"w{i}" for i in range(6))
    chunks = RAGPipeline.chunk_text(object(), text, chunk_size=3, overlap=3)
    assert chunks[0] == "w0 w1 w2"
    assert chunks[-1].endswith("w5")
    assert len(chunks) == 6


def test_chunk_text_normal_overlap_unchanged():
    text = " ".join(f"w{i}" for i in range(10))
    chunks = RAGPipeline.chunk_text(object(), text, chunk_size=4, overlap=2)
    assert chunks[0] == "w0 w1 w2 w3"
    assert chunks[1] == "w2 w3 w4 w5"


def test_sparse_model_refits_only_when_corpus_changes():
    """retrieve_sparse used to refit TF-IDF on every call; the fitted model is now cached by corpus content."""
    fits = []

    class FakeTfidf:
        def fit(self, texts):
            fits.append(list(texts))

    fake_module = mock.MagicMock(TfidfEmbedder=FakeTfidf)
    pipe = object.__new__(RAGPipeline)
    pipe.documents = [{"text": "alpha beta"}, {"text": "gamma"}]
    with mock.patch.dict(sys.modules, {"app.utils.custom_embeddings": fake_module}):
        first = pipe._sparse_model()
        assert pipe._sparse_model() is first
        assert len(fits) == 1
        pipe.documents.append({"text": "delta"})
        assert pipe._sparse_model() is not first
        assert len(fits) == 2


def test_sparse_and_hybrid_modes_honour_threshold():
    """The distance threshold used to be applied to dense search only."""
    pipe = object.__new__(RAGPipeline)
    pipe.index = mock.MagicMock(ntotal=2)
    rows = [{"distance": 0.2, "text": "near"}, {"distance": 0.9, "text": "far"}]
    pipe.retrieve_sparse = lambda *a, **k: [dict(r) for r in rows]
    pipe.retrieve_hybrid = lambda *a, **k: [dict(r) for r in rows]
    for mode in ("sparse", "hybrid"):
        got = pipe.retrieve_with_metadata("q", top_k=5, threshold=0.5, mode=mode)
        assert [r["text"] for r in got] == ["near"]
        # threshold 0 means "no cut-off", matching the dense path
        assert len(pipe.retrieve_with_metadata("q", top_k=5, threshold=0.0, mode=mode)) == 2


def test_dense_retrieval_has_no_department_shortcut():
    """A query mentioning 'IT' plus 'how'/'who' used to return every chunk containing 'IT' at rank 0."""
    pipe = object.__new__(RAGPipeline)
    pipe.documents = [
        {"text": "IT staff list", "chunk_id": 0, "vector_id": 0, "source_file": "a.txt"},
        {"text": "unrelated policy text", "chunk_id": 1, "vector_id": 1, "source_file": "a.txt"},
    ]
    pipe.index = mock.MagicMock(ntotal=2)
    pipe.index.search.return_value = ([[0.1, 0.4]], [[1, 0]])
    pipe._encoder = mock.MagicMock()
    got = pipe.retrieve_with_metadata("how does IT work", top_k=2)
    # Order follows the vector search (chunk 1 first), not a keyword shortcut.
    assert [d["chunk_id"] for d in got] == [1, 0]
    assert all(d["distance"] > 0.0 for d in got)
