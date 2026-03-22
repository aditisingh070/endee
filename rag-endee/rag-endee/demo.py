"""
demo.py – Run a quick end-to-end RAG demo using Endee vector database.

This demo:
  1. Starts a local Endee instance via Docker (if not already running)
  2. Ingests a small AI/ML knowledge base
  3. Runs several sample semantic queries
  4. Prints the retrieved context and assembled RAG prompt
"""

import json
from src.rag_pipeline import RAGPipeline

# ---------------------------------------------------------------------------
# Sample knowledge-base documents
# ---------------------------------------------------------------------------
SAMPLE_DOCUMENTS = [
    {
        "id": "doc_transformer",
        "title": "Transformer Architecture",
        "content": (
            "The Transformer is a deep learning model architecture introduced in the 2017 paper "
            "'Attention Is All You Need' by Vaswani et al. Unlike RNNs and LSTMs, the Transformer "
            "relies entirely on self-attention mechanisms to draw global dependencies between input "
            "and output. The architecture consists of an encoder and a decoder, each made up of "
            "stacked layers. Each encoder layer has two sub-layers: a multi-head self-attention "
            "mechanism and a position-wise fully connected feed-forward network. The decoder has "
            "an additional cross-attention layer. Transformers enable parallelization during "
            "training and have become the foundation of modern NLP models like BERT, GPT, and T5. "
            "The key innovation is the attention mechanism, which computes a weighted sum of "
            "values where the weights are determined by the compatibility of queries and keys. "
            "Positional encoding is added to the input embeddings to provide sequence order information."
        ),
    },
    {
        "id": "doc_rag",
        "title": "Retrieval Augmented Generation (RAG)",
        "content": (
            "Retrieval Augmented Generation (RAG) is a technique that combines information "
            "retrieval with language model generation. RAG was introduced by Lewis et al. (2020) "
            "at Facebook AI. The core idea is to retrieve relevant documents from a knowledge "
            "base and use them as additional context when generating a response. This grounds the "
            "language model in factual information and reduces hallucination. The RAG pipeline "
            "typically involves three stages: (1) Indexing – documents are split into chunks, "
            "embedded using an encoder model, and stored in a vector database. (2) Retrieval – "
            "at query time, the question is embedded and a nearest-neighbour search finds the "
            "most relevant chunks. (3) Generation – the retrieved chunks are concatenated with "
            "the question into a prompt, which is fed to a generative model to produce the answer. "
            "RAG is widely used in enterprise chatbots, question-answering systems, and "
            "knowledge management tools."
        ),
    },
    {
        "id": "doc_vector_db",
        "title": "Vector Databases",
        "content": (
            "A vector database is a type of database that stores data as high-dimensional vectors "
            "and supports approximate nearest-neighbour (ANN) search. Vector databases are central "
            "to modern AI applications including semantic search, recommendation systems, and "
            "retrieval augmented generation. Unlike traditional SQL databases that use exact "
            "matching, vector databases use indexing algorithms such as HNSW (Hierarchical "
            "Navigable Small World) or IVF (Inverted File Index) to find semantically similar "
            "vectors in milliseconds, even at billion-vector scale. Endee is a high-performance "
            "open-source vector database capable of handling up to 1 billion vectors on a single "
            "node. It supports cosine, dot-product, and Euclidean distance metrics, INT8 and "
            "FP16 precision, and provides both a Python SDK and a REST API. Popular alternatives "
            "include Pinecone, Weaviate, Qdrant, Chroma, Milvus, and Faiss."
        ),
    },
    {
        "id": "doc_embeddings",
        "title": "Sentence Embeddings",
        "content": (
            "Sentence embeddings are dense vector representations of sentences or short paragraphs "
            "that capture semantic meaning. Models like BERT, RoBERTa, and Sentence-Transformers "
            "encode text into fixed-size vectors (e.g. 384 or 768 dimensions) such that "
            "semantically similar sentences are close together in the vector space. The "
            "all-MiniLM-L6-v2 model from the Sentence Transformers library is a lightweight "
            "384-dimensional model widely used for semantic search tasks. It is fine-tuned on a "
            "large collection of sentence pairs using contrastive learning. Embeddings are the "
            "backbone of modern semantic search: a query and a document are both encoded, and the "
            "cosine similarity between their vectors determines relevance. Normalising embeddings "
            "to unit length allows cosine similarity to be computed as a simple dot product."
        ),
    },
    {
        "id": "doc_llm",
        "title": "Large Language Models",
        "content": (
            "Large Language Models (LLMs) are neural networks trained on massive text corpora to "
            "predict the next token in a sequence. GPT-4, Claude, Llama 3, and Gemini are "
            "prominent examples. These models contain billions of parameters and are pre-trained "
            "using self-supervised learning on diverse internet text. After pre-training, they "
            "are often fine-tuned with Reinforcement Learning from Human Feedback (RLHF) to "
            "follow instructions and align with human values. LLMs excel at text generation, "
            "summarisation, translation, code generation, and question answering. Their main "
            "weakness is hallucination – generating plausible-sounding but factually incorrect "
            "statements. Retrieval Augmented Generation (RAG) addresses this by providing "
            "retrieved factual context at inference time, grounding the model in real information."
        ),
    },
]

# ---------------------------------------------------------------------------
# Sample queries to test the pipeline
# ---------------------------------------------------------------------------
SAMPLE_QUERIES = [
    "How does the Transformer architecture work?",
    "What is Retrieval Augmented Generation?",
    "Why are vector databases used in AI applications?",
    "What embedding model should I use for semantic search?",
    "How does RAG reduce hallucination in large language models?",
]


def main():
    print("=" * 60)
    print("  RAG Pipeline Demo – powered by Endee Vector Database")
    print("=" * 60)

    # 1. Initialize pipeline
    pipeline = RAGPipeline()

    # 2. Ingest documents
    print("\n[1] Ingesting knowledge base ...")
    total = pipeline.ingest_documents(SAMPLE_DOCUMENTS)
    print(f"    ✓ {total} chunks stored in Endee index '{pipeline.index.name if hasattr(pipeline.index, 'name') else 'rag_knowledge_base'}'")

    # 3. Run queries
    print("\n[2] Running semantic queries ...\n")
    for q in SAMPLE_QUERIES:
        result = pipeline.query(q, top_k=3)

        print(f"Q: {result['question']}")
        print(f"   Latency : {result['latency_sec']}s")
        print("   Top context chunks:")
        for c in result["context"]:
            print(f"     • [{c['similarity']:.3f}] {c['title']} – {c['text'][:80]}...")
        print()

    # 4. Save last prompt to file (useful for feeding to an LLM)
    last_prompt_path = "last_rag_prompt.txt"
    with open(last_prompt_path, "w") as f:
        f.write(result["prompt"])
    print(f"[3] Last RAG prompt saved to: {last_prompt_path}")
    print("\nDemo complete. You can now feed the prompt to any LLM (OpenAI, Claude, Llama…).")


if __name__ == "__main__":
    main()
