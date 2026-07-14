from urllib.parse import urlsplit

from config import MODEL, OLLAMA_URL, QUERY_REWRITE_ENABLED, SKIP_MODEL_LOAD

try:
    import requests
except ModuleNotFoundError:
    if not SKIP_MODEL_LOAD:
        raise
    requests = None


def _ollama_base_url() -> str:
    parsed = urlsplit(OLLAMA_URL)
    return f"{parsed.scheme}://{parsed.netloc}"


def check_ollama_health(timeout: float = 3.0) -> dict:
    """檢查 Ollama 服務與目前設定的模型是否可用。"""
    if requests is None:
        return {
            "available": False,
            "model_available": False,
            "message": "requests 尚未載入，無法檢查 Ollama。",
        }

    try:
        response = requests.get(f"{_ollama_base_url()}/api/tags", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        return {
            "available": False,
            "model_available": False,
            "message": f"無法連線至 Ollama：{exc}",
        }
    except ValueError as exc:
        return {
            "available": False,
            "model_available": False,
            "message": f"Ollama 回傳無效資料：{exc}",
        }

    installed_models = {
        str(item.get("name", ""))
        for item in payload.get("models", [])
        if item.get("name")
    }
    model_available = MODEL in installed_models
    message = (
        f"Ollama 已連線，模型 {MODEL} 已安裝。"
        if model_available
        else f"Ollama 已連線，但找不到模型 {MODEL}；請先執行 ollama pull {MODEL}。"
    )
    return {
        "available": True,
        "model_available": model_available,
        "message": message,
        "installed_models": sorted(installed_models),
    }


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
4. 每個重要事實後必須標註支援它的來源編號，例如 [來源 1]。
5. 只能引用「檢索到的資料」中實際存在的來源編號，不得自行創造來源。
6. 若不同來源互相衝突，請明確指出衝突並分別引用。
7. 不要在回答內重複完整來源清單；程式會在回答後附上來源清單。
8. 對話紀錄只能作為理解上下文使用。
9. 最終回答仍必須以檢索內容為主要依據。
10. 如果檢索內容沒有答案，不能只靠記憶亂答。
11. 若問題是接續問題，例如「那它用什麼模型？」，可以參考最近對話補全主詞。

最近對話紀錄：
{history_text}

檢索到的資料：
{context}

問題：
{question}

回答：
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=300,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            "無法連線至 Ollama，請確認服務已啟動，並檢查設定的網址。"
        ) from exc

    response.raise_for_status()
    return response.json()["response"]


def rewrite_query(question: str, history: list[dict] | None = None) -> str:
    """將依賴前文的問題改寫成可獨立檢索的問題；失敗時回傳原問題。"""
    if not QUERY_REWRITE_ENABLED or not history or requests is None:
        return question

    recent_history = history[-3:]
    history_text = "\n\n".join(
        f"Q: {item.get('question', '')}\nA: {item.get('answer', '')}"
        for item in recent_history
    )
    prompt = f"""
你是 RAG 查詢改寫器。請根據最近對話，把本輪問題改寫成一個語意完整、
可獨立搜尋知識庫的繁體中文問題。

規則：
1. 只補全代名詞、省略主詞與必要術語。
2. 不回答問題，不新增對話中沒有的事實。
3. 若原問題已完整，原樣輸出。
4. 只輸出改寫後問題，不要加入解釋或引號。

最近對話：
{history_text}

本輪問題：
{question}
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        rewritten = str(response.json().get("response", "")).strip()
        return rewritten or question
    except (requests.RequestException, ValueError, KeyError):
        return question
