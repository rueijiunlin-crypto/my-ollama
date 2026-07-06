from config import MODEL, OLLAMA_URL, SKIP_MODEL_LOAD

try:
    import requests
except ModuleNotFoundError:
    if not SKIP_MODEL_LOAD:
        raise
    requests = None


def ask_ollama(question: str, context: str, history: list[dict] | None = None) -> str:
    if requests is None:
        raise RuntimeError("requests 尚未載入，無法呼叫 Ollama。")

    history_text = "無"
    if history:
        history_text = "\n\n".join(
            f"Q: {item.get('question', '')}\nA: {item.get('answer', '')}"
            for item in history
        )

    prompt = f"""
你是一個本地 RAG Agent。

規則：
1. 只能根據「檢索到的資料」回答。
2. 如果資料中沒有答案，請回答「資料中沒有提到」。
3. 請用繁體中文回答。
4. 回答時列出依據來源。
5. 對話紀錄只能作為理解上下文使用。
6. 最終回答仍必須以檢索內容為主要依據。
7. 如果檢索內容沒有答案，不能只靠記憶亂答。
8. 若問題是接續問題，例如「那它用什麼模型？」，可以參考最近對話補全主詞。

最近對話紀錄：
{history_text}

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
