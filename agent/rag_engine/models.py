import sys

import chromadb
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import (
    COLLECTION_NAME,
    DB_DIR,
    EMBEDDING_MODEL,
    RERANKER_ENABLED,
    RERANKER_MODEL,
    SKIP_MODEL_LOAD,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

embedding_model = None
reranker_model = None
client = None
collection = None


if not SKIP_MODEL_LOAD:
    try:
        print("載入 Embedding 模型：")
        print(EMBEDDING_MODEL)

        embedding_model = SentenceTransformer(
            EMBEDDING_MODEL,
            device=DEVICE,
        )

        print("Embedding 模型載入完成。")

    except Exception as exc:
        print("Embedding 模型載入失敗：")
        print(exc)
        sys.exit(1)

    if RERANKER_ENABLED:
        try:
            print("載入 Reranker 模型：")
            print(RERANKER_MODEL)

            reranker_model = CrossEncoder(
                RERANKER_MODEL,
                device=DEVICE,
            )

            print("Reranker 模型載入完成。")

        except Exception as exc:
            print("Reranker 模型載入失敗：")
            print(exc)
            print("將 fallback 回原本的向量搜尋結果。")

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

def print_runtime_info() -> None:
    print("\n=== 運算環境資訊 ===")
    print(f"PyTorch 版本：{torch.__version__}")
    print(f"CUDA 可用：{torch.cuda.is_available()}")
    print(f"PyTorch CUDA 版本：{torch.version.cuda}")
    print(f"目前運算裝置：{DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU：{torch.cuda.get_device_name(0)}")
        print(f"CUDA 裝置數量：{torch.cuda.device_count()}")
        print(f"目前 CUDA 裝置索引：{torch.cuda.current_device()}")

    if embedding_model is not None:
        print(f"Embedding 使用裝置：{embedding_model.device}")

    if reranker_model is not None:
        try:
            print(f"Reranker 使用裝置：{reranker_model.model.device}")
        except AttributeError:
            print(f"Reranker 使用裝置：{DEVICE}")

    print("====================\n")

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

if not SKIP_MODEL_LOAD:
    print_runtime_info()