import os

from dotenv import load_dotenv

load_dotenv()


# --------------------------------------------------
# Ollama
# --------------------------------------------------

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

OLLAMA_EMBEDDING_MODEL = os.getenv(
    "OLLAMA_EMBEDDING_MODEL",
    "nomic-embed-text",
)


# --------------------------------------------------
# Gemini
# --------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# --------------------------------------------------
# Chunking
# --------------------------------------------------

CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", "800")
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", "120")
)


# --------------------------------------------------
# Retrieval
# --------------------------------------------------

TOP_K_DENSE = int(
    os.getenv("TOP_K_DENSE", "10")
)

TOP_K_BM25 = int(
    os.getenv("TOP_K_BM25", "10")
)

TOP_K_HYBRID = int(
    os.getenv("TOP_K_HYBRID", "10")
)

TOP_K_RERANK = int(
    os.getenv("TOP_K_RERANK", "5")
)


# --------------------------------------------------
# Agent
# --------------------------------------------------

MAX_RETRIES = int(
    os.getenv("MAX_RETRIES", "2")
)


# --------------------------------------------------
# Storage
# --------------------------------------------------

CHROMA_PATH = "storage/chroma"

BM25_INDEX_PATH = "storage/indexes/bm25.pkl"