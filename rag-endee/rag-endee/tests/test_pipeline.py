"""
tests/test_pipeline.py – Unit tests for the RAG pipeline (no live Endee instance required).

Run with:  pytest tests/
"""

import sys, os, types
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub out heavy dependencies so tests run without GPU / Docker
# ---------------------------------------------------------------------------

# sentence_transformers stub
st_mod = types.ModuleType("sentence_transformers")
class _FakeEncoder:
    def encode(self, text, normalize_embeddings=False):
        import numpy as np
        return np.random.rand(384).astype("float32")
st_mod.SentenceTransformer = lambda *a, **kw: _FakeEncoder()
sys.modules.setdefault("sentence_transformers", st_mod)

# endee stub
endee_mod = types.ModuleType("endee")
class _FakePrecision:
    INT8 = "INT8"
endee_mod.Precision = _FakePrecision

class _FakeIndex:
    name = "rag_knowledge_base"
    def upsert(self, records): pass
    def query(self, vector, top_k=5):
        return [
            {"id": f"doc_{i}", "similarity": 0.9 - i*0.05,
             "meta": {"text": f"chunk {i}", "title": f"Title {i}",
                      "doc_id": f"d{i}", "chunk_index": 0, "total_chunks": 1}}
            for i in range(top_k)
        ]

class _FakeClient:
    def set_base_url(self, url): pass
    def list_indexes(self): return []
    def create_index(self, **kw): pass
    def get_index(self, name): return _FakeIndex()

endee_mod.Endee = lambda *a, **kw: _FakeClient()
sys.modules.setdefault("endee", endee_mod)

# Now we can safely import
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.rag_pipeline import chunk_text, RAGPipeline

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChunkText(unittest.TestCase):
    def test_short_text_single_chunk(self):
        chunks = chunk_text("hello world", chunk_size=300, overlap=50)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "hello world")

    def test_long_text_multiple_chunks(self):
        text   = " ".join([f"word{i}" for i in range(1000)])
        chunks = chunk_text(text, chunk_size=300, overlap=50)
        self.assertGreater(len(chunks), 1)

    def test_overlap(self):
        text   = " ".join([str(i) for i in range(600)])
        chunks = chunk_text(text, chunk_size=300, overlap=50)
        # Words at the boundary should appear in consecutive chunks
        boundary_end   = chunks[0].split()[-50:]
        boundary_start = chunks[1].split()[:50]
        self.assertEqual(boundary_end, boundary_start)

    def test_empty_text(self):
        self.assertEqual(chunk_text(""), [])


class TestRAGPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = RAGPipeline()

    def test_ingest_documents(self):
        docs = [
            {"id": "d1", "title": "Doc 1", "content": " ".join(["word"] * 100)},
        ]
        count = self.pipeline.ingest_documents(docs)
        self.assertEqual(count, 1)

    def test_retrieve_returns_list(self):
        results = self.pipeline.retrieve("test query", top_k=3)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIn("text", r)
            self.assertIn("similarity", r)

    def test_build_prompt_contains_question(self):
        context = [{"title": "T", "similarity": 0.9, "text": "Some context."}]
        prompt  = self.pipeline.build_prompt("What is AI?", context)
        self.assertIn("What is AI?", prompt)
        self.assertIn("Some context.", prompt)

    def test_query_structure(self):
        result = self.pipeline.query("What is a vector database?")
        self.assertIn("question", result)
        self.assertIn("context",  result)
        self.assertIn("prompt",   result)
        self.assertIn("answer",   result)
        self.assertIn("latency_sec", result)


if __name__ == "__main__":
    unittest.main()
