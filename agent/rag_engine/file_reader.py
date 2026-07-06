import json
from pathlib import Path
from typing import Any

from config import SUPPORTED_EXTENSIONS

try:
    from docx import Document
    from pypdf import PdfReader
except ModuleNotFoundError:
    Document = None
    PdfReader = None


def read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp950"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue

    return path.read_text(encoding="utf-8", errors="replace")


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: list[str] = []

    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text.strip())

    return "\n\n".join(parts)


def read_docx(path: Path) -> str:
    document = Document(path)
    parts: list[str] = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            parts.append(text)

    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    return "\n".join(parts)


def read_ipynb(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    parts: list[str] = []

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") not in {"markdown", "code"}:
            continue

        source = cell.get("source", "")
        if isinstance(source, list):
            text = "".join(source)
        else:
            text = str(source)

        if text.strip():
            parts.append(text.strip())

    return "\n\n".join(parts)


def _base_metadata(path: Path) -> dict[str, Any]:
    return {
        "source_path": str(path),
        "file_name": path.name,
        "file_type": path.suffix.lower(),
        "folder_name": path.parent.name,
    }


def _read_pdf_pages(path: Path, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    reader = PdfReader(str(path))
    docs: list[dict[str, Any]] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(
                {
                    "path": path,
                    "text": text.strip(),
                    "metadata": {
                        **metadata,
                        "page_number": page_number,
                    },
                }
            )

    if not docs:
        docs.append({"path": path, "text": "", "metadata": metadata})

    return docs


def read_files(folders: list[Path]) -> list[dict]:
    docs: list[dict] = []

    for folder in folders:
        if not folder.exists():
            print(f"找不到資料夾：{folder}")
            continue

        for path in folder.rglob("*"):
            if not path.is_file():
                continue

            suffix = path.suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                continue

            print(f"正在處理：{path}")
            metadata = _base_metadata(path)

            try:
                if suffix == ".pdf":
                    docs.extend(_read_pdf_pages(path, metadata))
                elif suffix == ".docx":
                    docs.append({"path": path, "text": read_docx(path), "metadata": metadata})
                elif suffix == ".ipynb":
                    docs.append({"path": path, "text": read_ipynb(path), "metadata": metadata})
                else:
                    docs.append({"path": path, "text": read_text_file(path), "metadata": metadata})
            except Exception as exc:
                print(f"讀取失敗：{path}")
                print(f"原因：{exc}")
                docs.append(
                    {
                        "path": path,
                        "text": "",
                        "metadata": {
                            **metadata,
                            "read_error": str(exc),
                        },
                    }
                )

    return docs


def read_single_file(path: Path) -> list[dict]:
    metadata = _base_metadata(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _read_pdf_pages(path, metadata)
    if suffix == ".docx":
        return [{"path": path, "text": read_docx(path), "metadata": metadata}]
    if suffix == ".ipynb":
        return [{"path": path, "text": read_ipynb(path), "metadata": metadata}]

    return [{"path": path, "text": read_text_file(path), "metadata": metadata}]
