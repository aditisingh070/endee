"""
RAG (Retrieval Augmented Generation) Pipeline using Endee Vector Database
=========================================================================
This module implements a complete RAG pipeline:
  1. Document ingestion and chunking
  2. Embedding generation (sentence-transformers)
  3. Vector storage in Endee
  4. Semantic retrieval
  5. Answer generation via OpenAI / HuggingFace (configurable)
"""

import os
import json
import time
import logging
from typing import List, Dict, Optional, Any

import numpy as np
from sentence_transformers import SentenceTransformer
from endee import Endee, Precision

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ENDEE_HOST      = os.getenv("ENDEE_HOST", "http://localhost:8080/api/v1")
ENDEE_TOKEN     = os.getenv("ENDEE_AUTH_TOKEN", "")          # leave blank for open mode
INDEX_NAME      = "rag_knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 384-dim, fast & lightweight
VECTOR_DIM      = 384
CHUNK_SIZE      = 300     # words per chunk
CHUNK_OVERLAP   = 50      # words overlap between consecutive chunks
TOP_K           = 5       # number of context chunks to retrieve


# ---------------------------------------------------------------------------
# Helper: Document Chunker
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split a long text into overlapping word-based chunks."""
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# RAG Pipeline class
# ---------------------------------------------------------------------------
class RAGPipeline:
    """
    End-to-end RAG pipeline backed by Endee vector database.

    Usage
    -----
    pipeline = RAGPipeline()
    pipeline.ingest_documents(documents)   # list of {"id", "title", "content"}
    answer   = pipeline.query("What is quantum computing?")
    """

    def __init__(self):
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        logger.info("Connecting to Endee at %s", ENDEE_HOST)
        self.client = Endee(ENDEE_TOKEN) if ENDEE_TOKEN else Endee()
        self.client.set_base_url(ENDEE_HOST)

        self._ensure_index()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------
    def _ensure_index(self):
        """Create the Endee index if it does not already exist."""
        existing = [idx["name"] for idx in self.client.list_indexes()]
        if INDEX_NAME not in existing:
            logger.info("Creating Endee index '%s' (dim=%d, cosine, INT8)", INDEX_NAME, VECTOR_DIM)
            self.client.create_index(
                name=INDEX_NAME,
                dimension=VECTOR_DIM,
                space_type="cosine",
                precision=Precision.INT8,
            )
        else:
            logger.info("Endee index '%s' already exists.", INDEX_NAME)

        self.index = self.client.get_index(name=INDEX_NAME)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Ingest a list of documents into Endee.

        Parameters
        ----------
        documents : list of dict
            Each dict must have keys: ``id``, ``title``, ``content``.

        Returns
        -------
        int  –  total number of chunks upserted
        """
        records   = []
        total_chunks = 0

        for doc in documents:
            doc_id  = doc["id"]
            title   = doc.get("title", "")
            content = doc["content"]

            chunks = chunk_text(content)
            logger.info("Document '%s': %d chunks", title, len(chunks))

            for i, chunk in enumerate(chunks):
                vector = self.embedder.encode(chunk, normalize_embeddings=True).tolist()
                records.append({
                    "id":     f"{doc_id}_chunk_{i}",
                    "vector": vector,
                    "meta": {
                        "doc_id":       doc_id,
                        "title":        title,
                        "chunk_index":  i,
                        "total_chunks": len(chunks),
                        "text":         chunk,
                    },
                })
                total_chunks += 1

                # Batch upsert every 64 records
                if len(records) >= 64:
                    self.index.upsert(records)
                    records = []

        # Flush remaining
        if records:
            self.index.upsert(records)

        logger.info("Ingestion complete. Total chunks stored: %d", total_chunks)
        return total_chunks

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """
        Retrieve the most relevant chunks for a given query.

        Returns
        -------
        list of dicts with keys: id, similarity, text, title, doc_id
        """
        query_vec = self.embedder.encode(query, normalize_embeddings=True).tolist()
        results   = self.index.query(vector=query_vec, top_k=top_k)

        context = []
        for r in results:
            context.append({
                "id":         r["id"],
                "similarity": round(r["similarity"], 4),
                "text":       r["meta"]["text"],
                "title":      r["meta"]["title"],
                "doc_id":     r["meta"]["doc_id"],
            })
        return context

    # ------------------------------------------------------------------
    # Generation  (prompt assembly – model-agnostic)
    # ------------------------------------------------------------------
    def build_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        """Assemble a RAG prompt from retrieved context chunks."""
        context_text = "\n\n".join(
            f"[Source: {c['title']}  |  Similarity: {c['similarity']}]\n{c['text']}"
            for c in context_chunks
        )
        prompt = (
            "You are a helpful AI assistant. Answer the question using ONLY the context below.\n"
            "If the answer is not in the context, say 'I don't have enough information.'\n\n"
            f"CONTEXT:\n{context_text}\n\n"
            f"QUESTION: {query}\n\nANSWER:"
        )
        return prompt

    # ------------------------------------------------------------------
    # End-to-end query
    # ------------------------------------------------------------------
    def query(self, question: str, top_k: int = TOP_K) -> Dict:
        """
        Full RAG query: retrieve context → build prompt → (optionally) generate answer.

        Returns a dict with:
          - question
          - context   (retrieved chunks)
          - prompt    (assembled RAG prompt ready for any LLM)
          - answer    (mock answer when no LLM key configured)
        """
        logger.info("Query: %s", question)
        t0 = time.time()

        context = self.retrieve(question, top_k=top_k)
        prompt  = self.build_prompt(question, context)

        # --- Optional: plug in your LLM here ---
        # from openai import OpenAI
        # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # response = client.chat.completions.create(
        #     model="gpt-4o-mini",
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # answer = response.choices[0].message.content

        # Fallback: return the assembled prompt so the caller can feed it to any LLM
        answer = (
            "[LLM not configured – set OPENAI_API_KEY and uncomment the OpenAI block above]\n\n"
            f"Top retrieved chunk:\n{context[0]['text'] if context else 'No results found.'}"
        )

        elapsed = round(time.time() - t0, 3)
        logger.info("Query completed in %.3fs", elapsed)

        return {
            "question":      question,
            "context":       context,
            "prompt":        prompt,
            "answer":        answer,
            "latency_sec":   elapsed,
        }
