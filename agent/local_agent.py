from pathlib import Path
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5-coder:14b"

DATA_DIR = Path("D:/AI_Server/knowledge_base")


def read_text_files(folder: Path) -> str:
    contents = []

    for path in folder.rglob("*"):
        if path.suffix.lower() in [
            ".txt",
            ".md",
            ".py",
            ".pdf",
            ".docx"
        ]:
            try:
                text = path.read_text(encoding="utf-8")
                contents.append(f"\n\n--- FILE: {path} ---\n{text}")
            except UnicodeDecodeError:
                pass

    return "\n".join(contents)


def ask_ollama(prompt: str) -> str:
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


def main():
    docs = read_text_files(DATA_DIR)

    print("===== 讀到的文件 =====")
    print(docs)
    print("====================")

    question = input("請輸入問題：")

    prompt = f"""
你是一個本地 AI Agent，請根據以下文件內容回答問題。
如果文件中沒有答案，請明確說「文件中沒有提到」。

文件內容：
{docs}

使用者問題：
{question}
"""

    answer = ask_ollama(prompt)
    print("\n=== Agent 回答 ===\n")
    print(answer)


if __name__ == "__main__":
    main()