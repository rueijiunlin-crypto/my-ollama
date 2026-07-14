import ast
import hashlib
import re
from typing import Any

from config import CHILD_CHUNK_OVERLAP, CHILD_CHUNK_SIZE, PARENT_CHUNK_SIZE


def split_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    text = text.strip()
    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap 必須小於 chunk_size")

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def _split_large_chunk(chunk: str, max_size: int = 1500) -> list[str]:
    if len(chunk) <= max_size:
        return [chunk.strip()] if chunk.strip() else []

    return split_text(chunk)


def split_markdown(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    heading_pattern = re.compile(r"(?m)^(#{1,4})\s+(.+?)\s*$")
    matches = list(heading_pattern.finditer(text))
    if not matches:
        return split_text(text)

    chunks: list[str] = []
    leading_text = text[: matches[0].start()].strip()
    if leading_text:
        chunks.extend(_split_large_chunk(leading_text))

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        chunks.extend(_split_large_chunk(section))

    return chunks


def split_python(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return split_text(text)

    lines = text.splitlines()
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and hasattr(node, "end_lineno")
    ]

    if not nodes:
        return split_text(text)

    chunks: list[str] = []
    covered_ranges: list[tuple[int, int]] = []

    for node in nodes:
        start_line = node.lineno
        if getattr(node, "decorator_list", None):
            start_line = min(decorator.lineno for decorator in node.decorator_list)

        end_line = node.end_lineno or node.lineno
        chunk = "\n".join(lines[start_line - 1 : end_line]).strip()
        if chunk:
            chunks.append(chunk)
            covered_ranges.append((start_line, end_line))

    extra_parts: list[str] = []
    cursor = 1
    for start_line, end_line in sorted(covered_ranges):
        if cursor < start_line:
            extra = "\n".join(lines[cursor - 1 : start_line - 1]).strip()
            if extra:
                extra_parts.append(extra)
        cursor = max(cursor, end_line + 1)

    if cursor <= len(lines):
        extra = "\n".join(lines[cursor - 1 :]).strip()
        if extra:
            extra_parts.append(extra)

    for extra in extra_parts:
        chunks.extend(split_text(extra))

    return chunks


def _find_matching_brace(text: str, open_brace_index: int) -> int | None:
    depth = 0
    in_string: str | None = None
    escaped = False

    for index in range(open_brace_index, len(text)):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == in_string:
                in_string = None
            continue

        if char in {'"', "'"}:
            in_string = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index

    return None


def split_cpp(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    pattern = re.compile(
        r"(?m)^[\t ]*(?:"
        r"(?:class|struct)\s+[A-Za-z_]\w*[^{;]*\{"
        r"|(?:[A-Za-z_][\w:<>,~*&\s]+)\s+[A-Za-z_]\w*\s*\([^;{}]*\)\s*(?:const\s*)?\{"
        r")"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return split_text(text)

    chunks: list[str] = []
    covered_ranges: list[tuple[int, int]] = []

    for match in matches:
        open_brace = text.find("{", match.start(), match.end())
        if open_brace == -1:
            continue

        close_brace = _find_matching_brace(text, open_brace)
        if close_brace is None:
            continue

        end = close_brace + 1
        if end < len(text) and text[end] == ";":
            end += 1

        chunk = text[match.start() : end].strip()
        if chunk:
            chunks.append(chunk)
            covered_ranges.append((match.start(), end))

    if not chunks:
        return split_text(text)

    extra_parts: list[str] = []
    cursor = 0
    for start, end in sorted(covered_ranges):
        if cursor < start:
            extra = text[cursor:start].strip()
            if extra:
                extra_parts.append(extra)
        cursor = max(cursor, end)

    if cursor < len(text):
        extra = text[cursor:].strip()
        if extra:
            extra_parts.append(extra)

    for extra in extra_parts:
        chunks.extend(split_text(extra))

    return chunks


def split_text_by_paragraph(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n+", text) if paragraph.strip()]
    if not paragraphs:
        return split_text(text)

    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > 1200:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(split_text(paragraph))
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= 1200:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())
        current = paragraph

    if current:
        chunks.append(current.strip())

    balanced_chunks: list[str] = []
    buffer = ""
    for chunk in chunks:
        candidate = f"{buffer}\n\n{chunk}".strip() if buffer else chunk
        if buffer and len(buffer) < 500 and len(candidate) <= 1200:
            buffer = candidate
        else:
            if buffer:
                balanced_chunks.append(buffer)
            buffer = chunk

    if buffer:
        balanced_chunks.append(buffer)

    return balanced_chunks


def _chunk_strategy(file_type: str, text: str) -> str:
    file_type = file_type.lower()

    if file_type == ".md" and re.search(r"(?m)^#{1,4}\s+", text):
        return "markdown_heading"
    if file_type == ".py":
        try:
            ast.parse(text)
            return "python_ast"
        except SyntaxError:
            return "fixed_window"
    if file_type in {".cpp", ".h", ".hpp"}:
        if re.search(r"(?m)^[\t ]*(?:class|struct)\s+[A-Za-z_]\w*|^[\t ]*(?:[A-Za-z_][\w:<>,~*&\s]+)\s+[A-Za-z_]\w*\s*\([^;{}]*\)\s*(?:const\s*)?\{", text):
            return "cpp_regex"
        return "fixed_window"
    if file_type in {".txt", ".docx", ".pdf", ".ipynb", ".json", ".yaml", ".yml"}:
        return "paragraph"
    return "fixed_window"


def chunk_document(text: str, file_type: str) -> list[str]:
    file_type = file_type.lower()

    if file_type == ".md":
        return split_markdown(text)
    if file_type == ".py":
        return split_python(text)
    if file_type in {".cpp", ".h", ".hpp"}:
        return split_cpp(text)
    if file_type in {".txt", ".docx", ".pdf", ".ipynb", ".json", ".yaml", ".yml"}:
        return split_text_by_paragraph(text)

    return split_text(text)


def create_parent_child_chunks(
    text: str,
    file_type: str,
    file_hash_prefix: str,
) -> list[dict[str, Any]]:
    """建立精準檢索用 Child 與完整回答用 Parent 的對應資料。"""
    structural_parents = chunk_document(text, file_type)
    parents: list[str] = []

    for parent in structural_parents:
        if len(parent) <= PARENT_CHUNK_SIZE:
            parents.append(parent)
        else:
            parents.extend(
                split_text(
                    parent,
                    chunk_size=PARENT_CHUNK_SIZE,
                    overlap=min(150, PARENT_CHUNK_SIZE // 5),
                )
            )

    records: list[dict[str, Any]] = []
    for parent_index, parent_text in enumerate(parents):
        parent_digest = hashlib.sha256(parent_text.encode("utf-8")).hexdigest()[:12]
        parent_id = f"{file_hash_prefix}_p{parent_index}_{parent_digest}"
        child_chunks = split_text(
            parent_text,
            chunk_size=CHILD_CHUNK_SIZE,
            overlap=CHILD_CHUNK_OVERLAP,
        )

        for child_index, child_text in enumerate(child_chunks):
            records.append(
                {
                    "parent_id": parent_id,
                    "parent_index": parent_index,
                    "parent_text": parent_text,
                    "child_index": child_index,
                    "child_text": child_text,
                }
            )

    return records


def _markdown_section_title(text_before_chunk: str) -> str | None:
    matches = re.findall(r"(?m)^\s{0,3}#{1,6}\s+(.+?)\s*$", text_before_chunk)
    return matches[-1].strip() if matches else None


def _python_function_or_class(chunk: str) -> str | None:
    try:
        tree = ast.parse(chunk)
    except SyntaxError:
        match = re.search(r"(?m)^\s*(?:async\s+def|def|class)\s+([A-Za-z_]\w*)", chunk)
        return match.group(1) if match else None

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return node.name

    return None


def _cpp_function_or_class(chunk: str) -> str | None:
    class_match = re.search(r"\b(?:class|struct)\s+([A-Za-z_]\w*)", chunk)
    if class_match:
        return class_match.group(1)

    function_match = re.search(
        r"\b(?:[A-Za-z_][\w:<>,~*&\s]+)\s+([A-Za-z_]\w*)\s*\([^;{}]*\)\s*(?:const\s*)?\{",
        chunk,
    )
    if function_match:
        return function_match.group(1)

    return None


def _chunk_metadata(
    doc: dict,
    chunk: str,
    chunk_id: str,
    start_index: int,
    chunk_strategy: str,
) -> dict[str, Any]:
    metadata = dict(doc["metadata"])
    metadata["chunk_id"] = chunk_id
    metadata["chunk_strategy"] = chunk_strategy

    file_type = metadata.get("file_type", "").lower()

    if file_type == ".md":
        section_title = _markdown_section_title(doc["text"][:start_index] + chunk)
        if section_title:
            metadata["section_title"] = section_title
    elif file_type == ".py":
        function_or_class = _python_function_or_class(chunk)
        if function_or_class:
            metadata["function_or_class"] = function_or_class
    elif file_type in {".cpp", ".h", ".hpp"}:
        function_or_class = _cpp_function_or_class(chunk)
        if function_or_class:
            metadata["function_or_class"] = function_or_class

    return metadata
