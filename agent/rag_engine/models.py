import sys

from config import (
    COLLECTION_NAME,
    DB_DIR,
    EMBEDDING_MODEL,
    RERANKER_ENABLED,
    RERANKER_MODEL,
    SKIP_MODEL_LOAD,
)

try:
    import chromadb
    from sentence_transformers import CrossEncoder, SentenceTransformer
except ModuleNotFoundError:
    if not SKIP_MODEL_LOAD:
        raise
    chromadb = None
    CrossEncoder = None
    SentenceTransformer = None

embedding_model = None
reranker_model = None
client = None
collection = None

if not SKIP_MODEL_LOAD:
    try:
        print("載入 Embedding 模型：")
        print(EMBEDDING_MODEL)
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print("Embedding 模型載入完成。")
    except Exception as exc:
        print("Embedding 模型載入失敗：")
        print(exc)
        sys.exit(1)

    if RERANKER_ENABLED:
        try:
            print("載入 Reranker 模型：")
            print(RERANKER_MODEL)
            reranker_model = CrossEncoder(RERANKER_MODEL)
            print("Reranker 模型載入完成。")
        except Exception as exc:
            print("Reranker 模型載入失敗：")
            print(exc)
            print("將 fallback 回原本的向量搜尋結果。")

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)


def get_embedding_model():
    return embedding_model


def get_reranker_model():
    return reranker_model


def get_collection():
    return collection


def encode_text(text: str) -> list[float]:
    model = get_embedding_model()
    if model is None:
        raise RuntimeError("Embedding 模型尚未載入。")

    return model.encode(
        text,
        normalize_embeddings=True,
    ).tolist()
