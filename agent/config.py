import os
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:14b"
# MODEL = "qwen2.5-coder:14b"

DATA_DIRS = [
    Path(r"G:\NKUST_VLM\learning"),
    Path(r"G:\AI_Server\knowledge_base"),
]

DB_DIR = r"G:\AI_Server\agent\chroma_db"
COLLECTION_NAME = "nkust_knowledge_bge_m3"
MANIFEST_PATH = Path(r"G:\AI_Server\agent\index_manifest.json")
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_ENABLED = True
BM25_ENABLED = True
BM25_TOP_K = 20
VECTOR_SEARCH_TOP_K = 20
RERANK_TOP_K = 5
BM25_WEIGHT = 0.25
RERANK_WEIGHT = 1.0
KEYWORD_WEIGHT = 0.15
CONVERSATION_MEMORY_SIZE = 5
SKIP_MODEL_LOAD = os.environ.get("RAG_AGENT_SKIP_MODEL_LOAD") == "1"

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".py",
    ".cpp",
    ".h",
    ".hpp",
    ".json",
    ".yaml",
    ".yml",
    ".ipynb",
    ".pdf",
    ".docx",
}
