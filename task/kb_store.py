"""
Knowledge-base ingestion: accept .txt/.md/.pdf/.docx, extract plain text,
save into the KB folder so the BM25 index can pick it up.
"""

from __future__ import annotations

import re
from pathlib import Path

ALLOWED_EXTS = {".txt", ".md", ".markdown", ".pdf", ".docx"}


def safe_name(name: str) -> str:
    base = name.strip().replace("\\", "/").split("/")[-1]
    base = re.sub(r"[^\w\-. ()]+", "_", base)
    base = base.strip(" .") or "upload"
    return base[:180]


def kind_of(name: str) -> str:
    n = name.lower()
    if n.endswith(".pdf"):
        return "pdf"
    if n.endswith(".docx"):
        return "docx"
    if n.endswith(".md") or n.endswith(".markdown"):
        return "md"
    return "txt"


def _extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError(
            "PDF support needs `pypdf`. Run: pip install -r requirements.txt"
        ) from e
    reader = PdfReader(str(path))
    out: list[str] = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        t = t.strip()
        if t:
            out.append(f"-- page {i + 1} --\n{t}")
    return "\n\n".join(out).strip()


def _extract_docx(path: Path) -> str:
    try:
        import docx  # python-docx
    except ImportError as e:
        raise RuntimeError(
            "DOCX support needs `python-docx`. Run: pip install -r requirements.txt"
        ) from e
    document = docx.Document(str(path))
    parts: list[str] = []
    for p in document.paragraphs:
        t = (p.text or "").rstrip()
        if t:
            parts.append(t)
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text and c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts).strip()


def extract_text(path: Path) -> str:
    k = kind_of(path.name)
    if k == "pdf":
        return _extract_pdf(path)
    if k == "docx":
        return _extract_docx(path)
    return path.read_text(encoding="utf-8", errors="replace")


def list_files(kb_root: Path) -> list[dict]:
    if not kb_root.exists():
        return []
    items: list[dict] = []
    for path in sorted(kb_root.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        items.append(
            {
                "name": path.name,
                "size": stat.st_size,
                "kind": kind_of(path.name),
            }
        )
    return items


def save_upload(kb_root: Path, file_storage) -> dict:
    """Save an uploaded file into the KB folder, extracting text if needed.

    .txt / .md are stored as-is.
    .pdf / .docx are stored as the original PLUS a sibling .txt with extracted text
    so the BM25 index picks them up. The .txt name is `<orig>.txt`.
    """
    name = safe_name(file_storage.filename or "upload.txt")
    if "." not in name:
        name = name + ".txt"
    ext = "." + name.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTS:
        raise RuntimeError(
            f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(ALLOWED_EXTS))}"
        )

    kb_root.mkdir(parents=True, exist_ok=True)
    final_name = name
    target = kb_root / final_name
    counter = 1
    stem, dot, suf = name.rpartition(".")
    while target.exists():
        final_name = f"{stem}_{counter}.{suf}" if dot else f"{name}_{counter}"
        target = kb_root / final_name
        counter += 1

    file_storage.save(str(target))

    chars = 0
    extracted_path: Path | None = None
    try:
        text = extract_text(target)
    except RuntimeError:
        # We saved the original; surface the dependency error to the caller.
        raise
    chars = len(text or "")

    if ext in (".pdf", ".docx"):
        # Companion .txt for indexing.
        out_path = target.with_suffix(target.suffix + ".txt")
        # Avoid clobbering an existing companion.
        c = 1
        while out_path.exists():
            out_path = target.with_name(target.name + f"_{c}.txt")
            c += 1
        out_path.write_text(text or "", encoding="utf-8")
        extracted_path = out_path

    return {
        "name": final_name,
        "kind": kind_of(final_name),
        "chars": chars,
        "indexed_as": (extracted_path.name if extracted_path else final_name),
    }


def delete_file(kb_root: Path, name: str) -> bool:
    name = safe_name(name)
    target = kb_root / name
    if not target.is_file() or name.startswith("."):
        return False
    target.unlink()
    # Remove companion extracted text, if any.
    companion = target.with_suffix(target.suffix + ".txt")
    if companion.is_file():
        try:
            companion.unlink()
        except OSError:
            pass
    return True
