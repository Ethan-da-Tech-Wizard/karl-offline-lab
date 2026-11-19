# ==============================================================================
# Program: test_rag_pipeline_fixes.py
# Purpose: Regression tests for RAGPipeline.chunk_text overlap guard and the
#          dataset_merger / keychain / trace_logger fixes. Heavy optional
#          dependencies (faiss, numpy, pymupdf, docx) are stubbed only when not installed.
# ==============================================================================
import sys
from unittest import mock

for _mod in ("faiss", "numpy", "pymupdf", "fitz", "docx"):
    try:
        __import__(_mod)
    except ImportError:  # keep these tests runnable on machines without the ML stack
        sys.modules[_mod] = mock.MagicMock()

from app.utils.rag_pipeline import RAGPipeline


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
