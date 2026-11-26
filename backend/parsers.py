"""Document parsers for course and faculty onboarding files.

Supports PDF, DOCX, PPTX, TXT, MD, CSV, JSON, and JSONL with UTF-8 Arabic and
English preservation and zero-dependency stdlib fallbacks for Office formats.
"""

from __future__ import annotations

import csv
import io
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".md", ".csv", ".json", ".jsonl"}

ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
LATIN_CHAR_RE = re.compile(r"[A-Za-z]")


@dataclass
class ExtractedSection:
    heading: str
    content: str
    location: str
    language: str


def detect_language(text: str) -> str:
    """Return 'ar', 'en', or 'bilingual' based on script character distribution."""
    ar_count = len(ARABIC_CHAR_RE.findall(text))
    en_count = len(LATIN_CHAR_RE.findall(text))
    total = ar_count + en_count
    if total == 0:
        return "en"
    ar_ratio = ar_count / total
    if ar_ratio >= 0.72:
        return "ar"
    if ar_ratio <= 0.20:
        return "en"
    return "bilingual"


def clean_text(text: str) -> str:
    """Normalize whitespace and stray control characters while keeping Arabic/English intact."""
    if not text:
        return ""
    text = text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _decode_bytes(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1256", "windows-1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _parse_pdf(raw: bytes) -> List[ExtractedSection]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF support requires pypdf to be installed.") from exc

    reader = PdfReader(io.BytesIO(raw))
    sections: List[ExtractedSection] = []
    for idx, page in enumerate(reader.pages, start=1):
        extracted = clean_text(page.extract_text() or "")
        if not extracted:
            continue
        first_line = extracted.split("\n", 1)[0][:90]
        sections.append(
            ExtractedSection(
                heading=first_line if len(first_line) < 85 else f"Page {idx}",
                content=extracted,
                location=f"Page {idx}",
                language=detect_language(extracted),
            )
        )
    return sections


def _parse_docx_stdlib(raw: bytes) -> List[ExtractedSection]:
    """Pure standard-library DOCX XML parser fallback."""
    sections: List[ExtractedSection] = []
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if "word/document.xml" not in archive.namelist():
            return sections
        xml_bytes = archive.read("word/document.xml")

    root = ET.fromstring(xml_bytes)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: List[str] = []
    for para in root.findall(".//w:p", ns):
        texts = [node.text for node in para.findall(".//w:t", ns) if node.text]
        joined = "".join(texts).strip()
        if joined:
            paragraphs.append(joined)

    return _split_plain_sections("\n\n".join(paragraphs), default_location="Document")


def _parse_docx(raw: bytes) -> List[ExtractedSection]:
    try:
        import docx  # type: ignore
    except ImportError:
        return _parse_docx_stdlib(raw)

    document = docx.Document(io.BytesIO(raw))
    blocks: List[Tuple[str, str]] = []
    current_heading = "Overview"
    current_lines: List[str] = []

    for para in document.paragraphs:
        text = clean_text(para.text)
        if not text:
            continue
        style_name = (para.style.name or "").lower() if para.style else ""
        if "heading" in style_name or "title" in style_name:
            if current_lines:
                blocks.append((current_heading, "\n".join(current_lines)))
                current_lines = []
            current_heading = text[:100]
        else:
            current_lines.append(text)

    for table in document.tables:
        for row in table.rows:
            cells = [clean_text(c.text) for c in row.cells if clean_text(c.text)]
            if cells:
                current_lines.append(" | ".join(cells))

    if current_lines:
        blocks.append((current_heading, "\n".join(current_lines)))

    sections: List[ExtractedSection] = []
    for idx, (heading, content) in enumerate(blocks, start=1):
        cleaned = clean_text(content)
        if cleaned:
            sections.append(
                ExtractedSection(
                    heading=heading,
                    content=cleaned,
                    location=f"Section {idx}",
                    language=detect_language(f"{heading} {cleaned}"),
                )
            )
    return sections


def _parse_pptx_stdlib(raw: bytes) -> List[ExtractedSection]:
    sections: List[ExtractedSection] = []
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        slide_names = sorted(
            [n for n in archive.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)],
            key=lambda name: int(re.search(r"(\d+)", name).group(1)),  # type: ignore[union-attr]
        )
        for idx, slide_name in enumerate(slide_names, start=1):
            root = ET.fromstring(archive.read(slide_name))
            texts = [node.text.strip() for node in root.iter() if node.tag.endswith("}t") and node.text and node.text.strip()]
            combined = clean_text("\n".join(texts))
            if combined:
                heading = texts[0][:90] if texts else f"Slide {idx}"
                sections.append(
                    ExtractedSection(
                        heading=heading,
                        content=combined,
                        location=f"Slide {idx}",
                        language=detect_language(combined),
                    )
                )
    return sections


def _parse_pptx(raw: bytes) -> List[ExtractedSection]:
    try:
        from pptx import Presentation  # type: ignore
    except ImportError:
        return _parse_pptx_stdlib(raw)

    prs = Presentation(io.BytesIO(raw))
    sections: List[ExtractedSection] = []
    for idx, slide in enumerate(prs.slides, start=1):
        lines: List[str] = []
        title_text = ""
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False):
                txt = clean_text(shape.text_frame.text)
                if txt:
                    if not title_text and shape == slide.shapes.title:
                        title_text = txt
                    lines.append(txt)
            elif getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    cells = [clean_text(cell.text) for cell in row.cells if clean_text(cell.text)]
                    if cells:
                        lines.append(" | ".join(cells))
        combined = clean_text("\n".join(lines))
        if combined:
            sections.append(
                ExtractedSection(
                    heading=(title_text or lines[0])[:95],
                    content=combined,
                    location=f"Slide {idx}",
                    language=detect_language(combined),
                )
            )
    return sections


def _split_plain_sections(text: str, default_location: str = "Section") -> List[ExtractedSection]:
    cleaned = clean_text(text)
    if not cleaned:
        return []

    heading_pattern = re.compile(r"^(?:#{1,4}\s+.+|[A-Za-z\u0600-\u06FF][^.\n]{2,75}:)$")
    sections: List[ExtractedSection] = []
    current_heading = "General Notes"
    buffer: List[str] = []
    section_index = 1

    for line in cleaned.split("\n"):
        stripped = line.strip()
        if heading_pattern.match(stripped) and not re.match(r"^(Q|A|Question|Answer|س|ج|سؤال|إجابة)\s*:", stripped, re.I):
            if buffer:
                body = clean_text("\n".join(buffer))
                if body:
                    sections.append(
                        ExtractedSection(
                            heading=current_heading,
                            content=body,
                            location=f"{default_location} {section_index}",
                            language=detect_language(f"{current_heading} {body}"),
                        )
                    )
                    section_index += 1
                buffer = []
            current_heading = re.sub(r"^#{1,4}\s*", "", stripped).rstrip(":").strip()
        else:
            buffer.append(line)

    if buffer:
        body = clean_text("\n".join(buffer))
        if body:
            sections.append(
                ExtractedSection(
                    heading=current_heading,
                    content=body,
                    location=f"{default_location} {section_index}",
                    language=detect_language(f"{current_heading} {body}"),
                )
            )
    return sections


def _parse_csv(raw: bytes) -> List[ExtractedSection]:
    decoded = _decode_bytes(raw)
    reader = csv.DictReader(io.StringIO(decoded))
    sections: List[ExtractedSection] = []
    if not reader.fieldnames:
        return _split_plain_sections(decoded, default_location="Row")

    for idx, row in enumerate(reader, start=1):
        pairs = [f"{(k or '').strip()}: {clean_text(str(v or ''))}" for k, v in row.items() if clean_text(str(v or ""))]
        if not pairs:
            continue
        content = "\n".join(pairs)
        first_val = next((clean_text(str(v)) for v in row.values() if clean_text(str(v))), f"Row {idx}")
        sections.append(
            ExtractedSection(
                heading=first_val[:90],
                content=content,
                location=f"Row {idx}",
                language=detect_language(content),
            )
        )
    return sections


def _parse_json_or_jsonl(raw: bytes, ext: str) -> List[ExtractedSection]:
    decoded = _decode_bytes(raw).strip()
    items = []
    if ext == ".jsonl":
        for line in decoded.splitlines():
            if line.strip():
                items.append(json.loads(line))
    else:
        parsed = json.loads(decoded)
        items = parsed if isinstance(parsed, list) else [parsed]

    sections: List[ExtractedSection] = []
    for idx, item in enumerate(items, start=1):
        if isinstance(item, dict):
            if "question" in item and "answer" in item:
                content = f"Question: {item['question']}\nAnswer: {item['answer']}"
                heading = str(item.get("topic") or item["question"])[:90]
            else:
                lines = [f"{k}: {v}" for k, v in item.items() if v not in (None, "", [])]
                content = clean_text("\n".join(lines))
                heading = str(item.get("title") or item.get("heading") or f"Record {idx}")[:90]
        else:
            content = clean_text(str(item))
            heading = f"Record {idx}"
        if content:
            sections.append(
                ExtractedSection(
                    heading=heading,
                    content=content,
                    location=f"Record {idx}",
                    language=detect_language(content),
                )
            )
    return sections


def parse_uploaded_file(filename: str, raw_bytes: bytes) -> List[ExtractedSection]:
    """Parse an uploaded course or faculty file into structured text sections."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type {ext}. Upload PDF, DOCX, PPTX, TXT, MD, CSV, or JSON.")

    if ext == ".pdf":
        return _parse_pdf(raw_bytes)
    if ext == ".docx":
        return _parse_docx(raw_bytes)
    if ext == ".pptx":
        return _parse_pptx(raw_bytes)
    if ext == ".csv":
        return _parse_csv(raw_bytes)
    if ext in {".json", ".jsonl"}:
        return _parse_json_or_jsonl(raw_bytes, ext)
    return _split_plain_sections(_decode_bytes(raw_bytes), default_location="Section")
