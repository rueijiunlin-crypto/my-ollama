from pathlib import Path
import requests
import chromadb
from sentence_transformers import SentenceTransformer

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5-coder:14b"

DATA_DIRS = [
    Path(r"G:\AI_Server\knowledge_base"),
]

DB_DIR = r"G:\AI_Server\agent\chroma_db"

embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(name="nkust_knowledge")


def split_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


def read_files(folders: list[Path]) -> list[tuple[str, str]]:
    docs = []

    for folder in folders:
        if not folder.exists():
            print(f"找不到資料夾：{folder}")
            continue

        for path in folder.rglob("*"):
            if path.suffix.lower() in [".txt", ".md", ".py"]:
                try:
                    docs.append((str(path), path.read_text(encoding="utf-8")))
                except UnicodeDecodeError:
                    print(f"讀取失敗：{path}")

    return docs


def clear_collection():
    try:
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass


def build_index():
    clear_collection()

    doc_id = 0
    files = read_files(DATA_DIRS)

    for file_path, text in files:
        chunks = split_text(text)

        for i, chunk in enumerate(chunks):
            embedding = embedding_model.encode(chunk).tolist()

            collection.add(
                ids=[f"doc_{doc_id}"],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[
                    {
                        "source": file_path,
                        "chunk": i,
                    }
                ],
            )

            doc_id += 1

    print(f"\n索引建立完成，共 {doc_id} 個 chunks。")


def search_docs(question: str, top_k: int = 8) -> str:
    query_embedding = embedding_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    contexts = []

    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        contexts.append(
            f"來源：{meta['source']}，chunk：{meta['chunk']}\n內容：{doc}"
        )

    return "\n\n---\n\n".join(contexts)


def ask_ollama(question: str, context: str) -> str:
    prompt = f"""
你是一個本地 RAG Agent。

規則：
1. 只能根據「檢索到的資料」回答。
2. 如果資料中沒有答案，請回答「資料中沒有提到」。
3. 請用繁體中文回答。
4. 回答時列出依據來源。

檢索到的資料：
{context}

問題：
{question}

回答：
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=300,
    )

    response.raise_for_status()
    return response.json()["response"]


def main() -> None:
    while True:
        print("\n=== NKUST Local RAG Agent v0.4 ===")
        print("1. 建立/重建索引")
        print("2. 進入常駐問答模式")
        print("q. 離開")

        mode = input("請選擇模式：").strip()

        if mode == "1":
            build_index()
            print("\n索引重建完畢，已回到主選單。")
            continue

        elif mode == "2":
            print("\n已進入常駐問答模式。")
            print("輸入 back 可回到主選單。")
            print("輸入 rebuild 可重建索引。")
            print("輸入 q 可離開。\n")

            while True:
                question = input("\n請輸入問題：").strip()

                if question.lower() == "q":
                    print("已離開 Agent。")
                    return

                if question.lower() == "back":
                    print("已回到主選單。")
                    break

                if question.lower() == "rebuild":
                    build_index()
                    print("\n索引重建完畢，回到問答模式。")
                    continue

                if not question:
                    continue

                context = search_docs(question)
                answer = ask_ollama(question, context)

                print("\n=== 檢索內容 ===\n")
                print(context)

                print("\n=== Agent 回答 ===\n")
                print(answer)

        elif mode.lower() == "q":
            print("已離開 Agent。")
            break

        else:
            print("無效模式，請重新選擇。")


if __name__ == "__main__":
    main()
