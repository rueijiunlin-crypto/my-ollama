import os
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:14b"
# MODEL = "qwen2.5-coder:14b"

DATA_DIRS = [
    Path(r"G:\AI_Server\knowledge_base"),
    Path(r"G:\NKUST_VLM\learning"),
]

DB_DIR = r"G:\AI_Server\agent\chroma_db"
COLLECTION_NAME = "nkust_knowledge_bge_m3"
MANIFEST_PATH = Path(r"G:\AI_Server\agent\index_manifest.json")
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_ENABLED = True
VECTOR_SEARCH_TOP_K = 20
RERANK_TOP_K = 5
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
