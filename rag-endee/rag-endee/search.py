"""
search.py – Interactive semantic search CLI using the Endee-backed RAG pipeline.

Usage:
    python search.py                    # interactive mode
    python search.py "your question"    # single query mode
"""

import sys
from src.rag_pipeline import RAGPipeline, TOP_K


def run_interactive(pipeline: RAGPipeline):
    print("\nEndee Semantic Search – type 'exit' to quit\n")
    while True:
        try:
            query = input("Search > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Bye!")
            break

        results = pipeline.retrieve(query, top_k=TOP_K)
        if not results:
            print("  No results found.\n")
            continue

        print(f"\n  Top {len(results)} results:\n")
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['similarity']:.4f}] {r['title']}")
            print(f"     {r['text'][:200]}...\n")


def main():
    pipeline = RAGPipeline()

    if len(sys.argv) > 1:
        query   = " ".join(sys.argv[1:])
        results = pipeline.retrieve(query)
        for i, r in enumerate(results, 1):
            print(f"{i}. [{r['similarity']:.4f}] {r['title']}\n   {r['text'][:300]}\n")
    else:
        run_interactive(pipeline)


if __name__ == "__main__":
    main()
