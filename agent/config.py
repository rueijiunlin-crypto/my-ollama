import os
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:14b"
# MODEL = "qwen2.5-coder:14b"

DATA_DIRS = [
    Path(r"G:\NKUST_VLM\learning"),
    Path(r"G:\AI_Server\knowledge_base"),
    Path(r"G:\AI_Server\README.md"),
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
KEYWORD_WEIGHT = 0.03
CONVERSATION_MEMORY_SIZE = 5
SKIP_MODEL_LOAD = os.environ.get("RAG_AGENT_SKIP_MODEL_LOAD") == "1"

# 檢索品質門檻。這些數值用於拒絕明顯缺乏證據的問題，之後可透過
# 真實問題集校準；不應將 final_score 當成機率或回答可信度。
MIN_RERANK_SCORE = 0.15
MIN_FALLBACK_FINAL_SCORE = 0.20
MIN_BM25_SCORE = 0.50
MIN_KEYWORD_SCORE = 0.15

# 第二輪：查詢、篩選與 Context 組裝。
QUERY_REWRITE_ENABLED = True
MAX_CHUNKS_PER_SOURCE = 2
CONTEXT_SIMILARITY_THRESHOLD = 0.85
MAX_CONTEXT_CHARACTERS = 8000
FILTERABLE_METADATA_FIELDS = {
    "root_source",
    "relative_path",
    "project",
    "module",
    "week",
    "file_type",
    "folder_name",
}

# 第三輪：批次索引。
EMBEDDING_BATCH_SIZE = 16
CHROMA_BATCH_SIZE = 64
INDEX_SCHEMA_VERSION = 2

# 第四輪：Parent-Child Retrieval。
PARENT_CHUNK_SIZE = 1800
CHILD_CHUNK_SIZE = 600
CHILD_CHUNK_OVERLAP = 100

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

# 索引掃描時一律略過的目錄名稱（不分大小寫）。
EXCLUDED_DIR_NAMES = {
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".git",
    ".vscode",
    ".idea",
    "site-packages",
    "chroma_db",
    "node_modules",
    "old_version",
    "old vesion",
}
