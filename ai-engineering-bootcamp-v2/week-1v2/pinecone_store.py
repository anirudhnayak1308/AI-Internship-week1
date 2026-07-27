"""Pinecone vector-store integration for the RAG assignment.

Embeddings and Pinecone are both configured entirely from environment variables -
no secrets live in this file. The same embedding model is used at ingest and query
time (embedding_query and embedding_upsert must match or similarity search breaks).

Run directly for a quick reachability check:
  python pinecone_store.py
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

THIS_DIR = Path(__file__).resolve().parent
load_dotenv(THIS_DIR / ".env")
load_dotenv(THIS_DIR.parent / ".env")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}

_openai_client: OpenAI | None = None
_pinecone_client: Pinecone | None = None
_index = None


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    """Embed one piece of text. Use this same function at ingest and query time."""
    response = get_openai_client().embeddings.create(model=model, input=text)
    return response.data[0].embedding


def get_pinecone_client() -> Pinecone:
    global _pinecone_client
    if _pinecone_client is None:
        _pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    return _pinecone_client


def get_index_name() -> str:
    return os.environ["PINECONE_INDEX_NAME"]


def ensure_index_exists() -> None:
    """Create the configured serverless index if it doesn't exist yet."""
    pc = get_pinecone_client()
    index_name = get_index_name()
    if index_name in pc.list_indexes().names():
        return
    pc.create_index(
        name=index_name,
        dimension=EMBEDDING_DIMENSIONS.get(EMBEDDING_MODEL, 1536),
        metric="cosine",
        spec=ServerlessSpec(
            cloud=os.getenv("PINECONE_CLOUD", "aws"),
            region=os.getenv("PINECONE_REGION", "us-east-1"),
        ),
    )


def get_index():
    global _index
    if _index is None:
        _index = get_pinecone_client().Index(get_index_name())
    return _index


def upsert_texts(items: list[tuple[str, str, dict]]) -> int:
    """Embed and upsert (id, text, metadata) triples. Returns the number upserted."""
    vectors = [
        {
            "id": vector_id,
            "values": get_embedding(text),
            "metadata": {**metadata, "text": text},
        }
        for vector_id, text, metadata in items
    ]
    get_index().upsert(vectors=vectors)
    return len(vectors)


def query_similar(text: str, top_k: int = 5, filter: dict | None = None):
    """Embed `text` with the same model used at ingest time and return top-k matches."""
    embedding = get_embedding(text)
    return get_index().query(
        vector=embedding, top_k=top_k, include_metadata=True, filter=filter
    )


def check_health() -> dict:
    """Confirm Pinecone is reachable and the configured index exists, without touching data."""
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not api_key:
        return {"ok": False, "error": "PINECONE_API_KEY is not set"}
    if not index_name:
        return {"ok": False, "error": "PINECONE_INDEX_NAME is not set"}

    try:
        pc = Pinecone(api_key=api_key)
        available = pc.list_indexes().names()
        if index_name not in available:
            return {
                "ok": False,
                "index_name": index_name,
                "error": f"Index '{index_name}' not found. Available: {available}",
            }
        stats = pc.Index(index_name).describe_index_stats()
        stats_dict = stats.to_dict() if hasattr(stats, "to_dict") else dict(stats)
        return {
            "ok": True,
            "index_name": index_name,
            "total_vector_count": stats_dict.get("total_vector_count"),
            "dimension": stats_dict.get("dimension"),
        }
    except Exception as exc:
        return {"ok": False, "index_name": index_name, "error": str(exc)}


if __name__ == "__main__":
    print(json.dumps(check_health(), indent=2))
