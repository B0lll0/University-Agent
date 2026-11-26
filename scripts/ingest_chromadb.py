"""Standalone Raspberry Pi ingestion script for ChromaDB + Ollama nomic-embed-text.

Usage on Raspberry Pi 3/4/5 or local workstation:
    pip install chromadb requests
    ollama pull nomic-embed-text
    python ingest_chromadb.py --input rag_chunks_chromadb.json --db ./chroma_db
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests


def get_embedding(text: str, ollama_url: str, model: str) -> list[float]:
    resp = requests.post(
        f"{ollama_url}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest exported campus RAG chunks into local ChromaDB.")
    parser.add_argument("--input", default="rag_chunks_chromadb.json", help="Path to exported JSON file.")
    parser.add_argument("--db", default="./chroma_db", help="Persistent ChromaDB directory.")
    parser.add_argument("--ollama", default="http://127.0.0.1:11434", help="Local Ollama base URL.")
    parser.add_argument("--model", default="nomic-embed-text", help="Ollama embedding model.")
    args = parser.parse_args()

    try:
        import chromadb  # type: ignore
    except ImportError:
        print("Error: chromadb is not installed. Run: pip install chromadb", file=sys.stderr)
        sys.exit(1)

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    ids = payload["ids"]
    documents = payload["documents"]
    metadatas = payload["metadatas"]
    collection_name = payload.get("collection_name", "campus_assistant_knowledge")

    print(f"Generating embeddings for {len(documents)} chunks using {args.model}...")
    embeddings = [get_embedding(doc, args.ollama, args.model) for doc in documents]

    client = chromadb.PersistentClient(path=args.db)
    collection = client.get_or_create_collection(name=collection_name)
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    print(f"Done. Stored {len(documents)} chunks in {args.db} ({collection_name}).")


if __name__ == "__main__":
    main()
