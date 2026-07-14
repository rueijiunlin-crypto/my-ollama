def _display_path(metadata: dict) -> str:
    source_path = str(metadata.get("source_path", "")).strip()
    if source_path:
        return source_path
    return str(metadata.get("file_name", "未知來源")).strip() or "未知來源"


def citation_label(index: int) -> str:
    return f"來源 {index}"


def format_source_list(results: list[dict]) -> str:
    """建立由程式產生、順序穩定的來源清單。"""
    if not results:
        return ""

    lines = ["參考來源："]
    for index, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        location: list[str] = []

        if metadata.get("section_title"):
            location.append(f"章節：{metadata['section_title']}")
        if metadata.get("page_number") is not None:
            location.append(f"頁碼：{metadata['page_number']}")
        if metadata.get("function_or_class"):
            location.append(f"函式/類別：{metadata['function_or_class']}")

        path = _display_path(metadata)
        location_text = f"（{'；'.join(location)}）" if location else ""
        lines.append(f"[{citation_label(index)}] {path}{location_text}")

    return "\n".join(lines)


def append_source_list(answer: str, results: list[dict]) -> str:
    """無論模型是否自行列出來源，都附上程式生成的可追溯來源清單。"""
    sources = format_source_list(results)
    if not sources:
        return answer
    return f"{answer.rstrip()}\n\n---\n{sources}"
