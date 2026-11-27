"""Dataset builder, bilingual Q&A extractor, local RAG retriever, and exporters.

Designed for the Offline Edge AI Campus Assistant (Raspberry Pi + Ollama + ChromaDB).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from backend.msa_curricula import (
    MSA_FACULTIES_DATA,
    MSA_FACULTY_SCHOLARSHIP_PARENTS,
    MSA_TUITION_FEES,
    TOP_5_MSA_FAQS,
    build_study_plans_markdown,
)
from backend.parsers import ExtractedSection, clean_text, detect_language, parse_uploaded_file

TOKEN_RE = re.compile(r"[\u0600-\u06FF]+|[A-Za-z0-9]+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?؟\n])\s+")


def estimate_tokens(text: str) -> int:
    """Approximate token count suitable for small edge LLMs."""
    words = TOKEN_RE.findall(text)
    return max(1, int(round(len(words) * 1.35)))


def _make_id(prefix: str, seed: str) -> str:
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def chunk_sections(
    sections: List[ExtractedSection],
    source_file: str,
    course_name: str,
    department: str,
    chunk_size: int = 512,
    chunk_overlap: int = 96,
) -> List[Dict[str, Any]]:
    """Split extracted sections into overlapping RAG chunks with ChromaDB metadata."""
    chunk_size = max(180, min(2000, int(chunk_size)))
    chunk_overlap = max(0, min(chunk_size // 2, int(chunk_overlap)))
    now_iso = datetime.now(timezone.utc).isoformat()
    chunks: List[Dict[str, Any]] = []

    for sec_idx, sec in enumerate(sections, start=1):
        full_text = clean_text(sec.content)
        if not full_text:
            continue

        pieces: List[str] = []
        if len(full_text) <= chunk_size:
            pieces.append(full_text)
        else:
            sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(full_text) if s.strip()]
            current = ""
            for sentence in sentences:
                if len(sentence) > chunk_size:
                    if current:
                        pieces.append(current.strip())
                        current = ""
                    step = max(80, chunk_size - chunk_overlap)
                    for start in range(0, len(sentence), step):
                        sub = sentence[start : start + chunk_size].strip()
                        if sub:
                            pieces.append(sub)
                    continue

                candidate = f"{current} {sentence}".strip() if current else sentence
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current:
                        pieces.append(current.strip())
                    tail = current[-chunk_overlap:].strip() if chunk_overlap > 0 and current else ""
                    current = f"{tail} {sentence}".strip() if tail else sentence

            if current:
                pieces.append(current.strip())

        for piece_idx, text in enumerate(pieces, start=1):
            chunk_lang = detect_language(text)
            chunk_id = _make_id("chk", f"{source_file}:{sec_idx}:{piece_idx}:{text[:60]}")
            chunks.append(
                {
                    "id": chunk_id,
                    "text": text,
                    "heading": sec.heading,
                    "source_file": source_file,
                    "location": sec.location,
                    "course": course_name,
                    "department": department,
                    "language": chunk_lang,
                    "char_count": len(text),
                    "token_estimate": estimate_tokens(text),
                    "created_at": now_iso,
                }
            )

    return chunks


def _extract_explicit_qa(text: str) -> List[Tuple[str, str]]:
    """Extract explicit Q/A pairs in English or Arabic from text."""
    pairs: List[Tuple[str, str]] = []
    pattern = re.compile(
        r"(?:^|\n)(?:Q|Question|س|سؤال)\s*[:：]\s*(.+?)\n(?:A|Answer|ج|إجابة|الاجابة)\s*[:：]\s*(.+?)(?=\n(?:Q|Question|س|سؤال)\s*[:：]|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(text):
        q = clean_text(match.group(1))
        a = clean_text(match.group(2))
        if len(q) >= 6 and len(a) >= 6:
            pairs.append((q, a))

    if not pairs:
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        idx = 0
        while idx < len(lines) - 1:
            line = lines[idx]
            if (line.endswith("?") or line.endswith("؟")) and len(line) >= 8:
                ans_lines = []
                j = idx + 1
                while j < len(lines) and not (lines[j].endswith("?") or lines[j].endswith("؟")):
                    ans_lines.append(lines[j])
                    j += 1
                    if len(" ".join(ans_lines)) > 420:
                        break
                ans = clean_text(" ".join(ans_lines))
                if len(ans) >= 8:
                    pairs.append((line, ans))
                idx = j
            else:
                idx += 1
    return pairs


def _synthesize_grounded_qa(chunk: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """Create a grounded Q&A pair directly from a chunk's heading and body."""
    heading = clean_text(chunk.get("heading", ""))
    body = clean_text(chunk.get("text", ""))
    if len(body) < 28:
        return None

    lang = chunk.get("language", "en")
    course = chunk.get("course", "").strip()
    generic_headings = {"general notes", "overview", "document", "page 1", "section 1"}
    is_generic = heading.lower() in generic_headings or re.match(r"^(page|slide|row|record|section)\s+\d+$", heading, re.I)

    if is_generic:
        first_sentence = SENTENCE_SPLIT_RE.split(body)[0].strip()
        topic = first_sentence[:72].rstrip(".:،")
        if lang == "ar":
            question = f"ما هي التفاصيل المذكورة حول {topic}؟"
        else:
            question = f"What key details are covered regarding {topic}?"
    else:
        if heading.endswith("?") or heading.endswith("؟"):
            question = heading
        elif lang == "ar":
            if course and course not in heading:
                question = f"ما هي المعلومات الأساسية حول {heading} في {course}؟"
            else:
                question = f"ما هي المعلومات الأساسية حول {heading}؟"
        else:
            if course and course.lower() not in heading.lower():
                question = f"What should students know about {heading} in {course}?"
            else:
                question = f"What should students know about {heading}?"

    return question, body


def _try_ollama_qa(chunk: Dict[str, Any], model: str, ollama_url: str = "http://127.0.0.1:11434") -> List[Tuple[str, str]]:
    """Optionally generate grounded Q&A pairs via local Ollama LLM."""
    prompt = (
        "You are preparing an offline campus assistant dataset. "
        "Based strictly on the passage below, write 2 concise question and answer pairs "
        "in the same language as the passage. Return valid JSON array of objects with keys "
        "\"question\" and \"answer\". Do not invent facts.\n\n"
        f"Heading: {chunk['heading']}\nPassage:\n{chunk['text']}"
    )
    try:
        resp = requests.post(
            f"{ollama_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "format": "json"},
            timeout=15,
        )
        if resp.status_code != 200:
            return []
        raw = resp.json().get("response", "")
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            parsed = parsed.get("pairs") or parsed.get("items") or [parsed]
        results: List[Tuple[str, str]] = []
        for item in parsed:
            if isinstance(item, dict) and item.get("question") and item.get("answer"):
                results.append((clean_text(str(item["question"])), clean_text(str(item["answer"]))))
        return results[:3]
    except Exception:
        return []


def extract_qa_pairs(
    chunks: List[Dict[str, Any]],
    use_ollama: bool = False,
    ollama_model: str = "phi3:mini",
) -> Tuple[List[Dict[str, Any]], str]:
    """Build bilingual grounded Q&A pairs from chunks."""
    now_iso = datetime.now(timezone.utc).isoformat()
    qa_pairs: List[Dict[str, Any]] = []
    seen_questions = set()
    engine_used = "offline_heuristic"

    ollama_ready = False
    if use_ollama:
        ollama_ready = check_ollama_status().get("available", False)
        if ollama_ready:
            engine_used = f"ollama:{ollama_model}"

    for chunk in chunks:
        candidates: List[Tuple[str, str]] = []
        if ollama_ready:
            candidates = _try_ollama_qa(chunk, model=ollama_model)
        if not candidates:
            candidates = _extract_explicit_qa(chunk["text"])
        if not candidates:
            synthesized = _synthesize_grounded_qa(chunk)
            if synthesized:
                candidates = [synthesized]

        for q, a in candidates:
            norm_q = q.lower().strip()
            if norm_q in seen_questions:
                continue
            seen_questions.add(norm_q)
            qa_id = _make_id("qa", f"{chunk['id']}:{norm_q[:80]}")
            qa_pairs.append(
                {
                    "id": qa_id,
                    "question": q,
                    "answer": a,
                    "topic": chunk["heading"],
                    "source_file": chunk["source_file"],
                    "chunk_id": chunk["id"],
                    "course": chunk["course"],
                    "department": chunk["department"],
                    "language": detect_language(f"{q} {a}"),
                    "created_at": now_iso,
                }
            )

    return qa_pairs, engine_used


def normalize_arabic(token: str) -> str:
    """Light Arabic normalization and prefix stripping for Egyptian and Standard Arabic search."""
    token = re.sub(r"[\u064B-\u065F\u0670]", "", token)
    token = token.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    token = token.replace("ى", "ي").replace("ة", "ه")
    token = token.lower()
    if re.match(r"^[\u0600-\u06FF]{4,}$", token):
        for prefix in ("وال", "بال", "كال", "فال", "لل", "ال"):
            if token.startswith(prefix) and len(token) - len(prefix) >= 3:
                token = token[len(prefix) :]
                break
    return token


def _tokenize(text: str) -> List[str]:
    keep_short = {"g", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
    return [
        normalize_arabic(tok)
        for tok in TOKEN_RE.findall(text)
        if len(tok) > 1 or tok.lower() in keep_short
    ]


MSA_BUILDINGS: Dict[str, Dict[str, str]] = {
    "A": {
        "en_name": "Building A",
        "ar_name": "مبنى A",
        "en_desc": "Houses University Administration, Faculty of Languages, Faculty of Mass Communication, and Faculty of Management Sciences.",
        "ar_desc": "يضم الإدارة الجامعية وكليات اللغات والإعلام والعلوم الإدارية.",
    },
    "1": {
        "en_name": "Building A",
        "ar_name": "مبنى A",
        "en_desc": "Houses University Administration, Faculty of Languages, Faculty of Mass Communication, and Faculty of Management Sciences.",
        "ar_desc": "يضم الإدارة الجامعية وكليات اللغات والإعلام والعلوم الإدارية.",
    },
    "B": {
        "en_name": "Building B",
        "ar_name": "مبنى B",
        "en_desc": "Serves as the Student Services Building (SSB) housing Student Affairs, registration support, student activities, seminar rooms, career events, and the campus Cinema Hall.",
        "ar_desc": "مبنى خدمات الطلاب (SSB) ويضم شؤون الطلاب والأنشطة الطلابية وقاعات الندوات وقاعة السينما الجامعية.",
    },
    "2": {
        "en_name": "Building B",
        "ar_name": "مبنى B",
        "en_desc": "Serves as the Student Services Building (SSB) housing Student Affairs, registration support, student activities, seminar rooms, career events, and the campus Cinema Hall.",
        "ar_desc": "مبنى خدمات الطلاب (SSB) ويضم شؤون الطلاب والأنشطة الطلابية وقاعات الندوات وقاعة السينما الجامعية.",
    },
    "SSB": {
        "en_name": "Building B (SSB — Student Services Building)",
        "ar_name": "مبنى B (مبنى خدمات الطلاب SSB)",
        "en_desc": "Central student hub for Student Affairs, registration services, student activities, seminar rooms, career fairs, and the renovated campus Cinema Hall.",
        "ar_desc": "المركز الرئيسي لشؤون الطلاب والأنشطة الطلابية والندوات وفعاليات التوظيف وقاعة السينما الجامعية.",
    },
    "C": {
        "en_name": "Building C",
        "ar_name": "مبنى C",
        "en_desc": "Main scientific academic complex housing Faculty of Pharmacy, Faculty of Engineering, Faculty of Biotechnology, Faculty of Computer Science, and Faculty of Dentistry.",
        "ar_desc": "المجمع الأكاديمي العلمي الرئيسي ويضم كليات الصيدلة والهندسة والتكنولوجيا الحيوية وعلوم الحاسب وطب الأسنان.",
    },
    "3": {
        "en_name": "Building C",
        "ar_name": "مبنى C",
        "en_desc": "Main scientific academic complex housing Faculty of Pharmacy, Faculty of Engineering, Faculty of Biotechnology, Faculty of Computer Science, and Faculty of Dentistry.",
        "ar_desc": "المجمع الأكاديمي العلمي الرئيسي ويضم كليات الصيدلة والهندسة والتكنولوجيا الحيوية وعلوم الحاسب وطب الأسنان.",
    },
    "D": {
        "en_name": "Building D",
        "ar_name": "مبنى D",
        "en_desc": "Dedicated to specialized scientific research laboratories and Faculty of Dentistry Outpatient Clinics.",
        "ar_desc": "مخصص للمعامل العلمية المتخصصة وعيادات كلية طب الأسنان الخارجية.",
    },
    "4": {
        "en_name": "Building D",
        "ar_name": "مبنى D",
        "en_desc": "Dedicated to specialized scientific research laboratories and Faculty of Dentistry Outpatient Clinics.",
        "ar_desc": "مخصص للمعامل العلمية المتخصصة وعيادات كلية طب الأسنان الخارجية.",
    },
    "E": {
        "en_name": "Building E",
        "ar_name": "مبنى E",
        "en_desc": "Houses the Faculty of Arts and Design (Interior Architecture, Graphics, and Media Arts), classrooms, and faculty offices.",
        "ar_desc": "يضم كلية الفنون والتصميم (العمارة الداخلية والجرافيك والفنون الإعلامية) وقاعات دراسية ومكاتب إدارية.",
    },
    "5": {
        "en_name": "Building E",
        "ar_name": "مبنى E",
        "en_desc": "Houses the Faculty of Arts and Design (Interior Architecture, Graphics, and Media Arts), classrooms, and faculty offices.",
        "ar_desc": "يضم كلية الفنون والتصميم (العمارة الداخلية والجرافيك والفنون الإعلامية) وقاعات دراسية ومكاتب إدارية.",
    },
    "F": {
        "en_name": "Building F",
        "ar_name": "مبنى F",
        "en_desc": "Part of the academic workshops and engineering support zone for Engineering and Design technical workshops.",
        "ar_desc": "يقع ضمن نطاق الورش الفنية والهندسية وورش الهندسة والتصميم.",
    },
    "6": {
        "en_name": "Building F",
        "ar_name": "مبنى F",
        "en_desc": "Houses Engineering and Design technical workshops and hands on fabrication spaces.",
        "ar_desc": "يضم ورش الهندسة والتصميم العملية.",
    },
    "7": {
        "en_name": "Building 7",
        "ar_name": "مبنى 7",
        "en_desc": "Houses the Main University Library and the MSA Gymnasium and fitness complex.",
        "ar_desc": "يضم المكتبة الجامعية الرئيسية وجيم الجامعة والمرافق الرياضية.",
    },
    "G": {
        "en_name": "Building G",
        "ar_name": "مبنى G",
        "en_desc": "Major campus landmark housing lecture halls, auditoriums, computer labs, staff rooms, Arts and Design exhibition spaces, cafeteria, and the 1,150 seat MSA Opera House indoor theater.",
        "ar_desc": "معلم معماري رئيسي يضم قاعات المحاضرات والمدرجات ومعامل الكمبيوتر ومكاتب هيئة التدريس ومعارض الفنون ودار أوبرا جامعة MSA بسعة 1150 مقعداً.",
    },
    "H": {
        "en_name": "Building H",
        "ar_name": "مبنى H",
        "en_desc": "Serves as the Research Center featuring dental outpatient clinics, operation theaters, radiology rooms, sterilization units, and computer labs.",
        "ar_desc": "مركز الأبحاث ويضم عيادات طب الأسنان وغرف العمليات والأشعة والتعقيم ومعامل الكمبيوتر.",
    },
    "K": {
        "en_name": "Building K",
        "ar_name": "مبنى K",
        "en_desc": "Hosts lecture halls, computer labs, and sessions for Computer Science and Pharmacy, plus Room K108 (official MSA Bus Subscription Room).",
        "ar_desc": "يضم قاعات محاضرات ومعامل لكلية علوم الحاسب وكلية الصيدلة، بالإضافة إلى غرفة K108 (مكتب اشتراكات أتوبيسات الجامعة).",
    },
    "L": {
        "en_name": "Building L",
        "ar_name": "مبنى L",
        "en_desc": "Academic classrooms and faculty support block inside the 50 acre MSA 6th of October campus.",
        "ar_desc": "مبنى قاعات دراسية وخدمات أكاديمية داخل حرم جامعة MSA في مدينة 6 أكتوبر.",
    },
    "M": {
        "en_name": "Building M",
        "ar_name": "مبنى M",
        "en_desc": "Houses specialized rooms and clinical workshops primarily used by the Faculty of Physical Therapy (including Room M108).",
        "ar_desc": "يضم القاعات وورش التدريب الإكلينيكي التابعة لكلية العلاج الطبيعي (بما في ذلك قاعة M108).",
    },
    "N": {
        "en_name": "Building N",
        "ar_name": "مبنى N",
        "en_desc": "Academic classrooms and laboratories block inside MSA University's 6th of October campus.",
        "ar_desc": "مبنى قاعات دراسية ومعامل داخل حرم جامعة MSA في مدينة 6 أكتوبر.",
    },
    "11": {
        "en_name": "Building 11",
        "ar_name": "مبنى 11",
        "en_desc": "Dedicated extension building for the Faculty of Oral and Dental Medicine.",
        "ar_desc": "مبنى التوسعة المخصص لكلية طب الفم والأسنان.",
    },
    "12": {
        "en_name": "Building 12",
        "ar_name": "مبنى 12",
        "en_desc": "Houses the Main Theater and large conference and seminar halls.",
        "ar_desc": "يضم المسرح الرئيسي وقاعات المؤتمرات والندوات الكبرى.",
    },
}

MSA_SPECIAL_ROOMS: Dict[str, Tuple[str, str]] = {
    "K108": (
        "Room K108 is the official MSA University Bus Room where students register and manage campus bus subscriptions.",
        "غرفة K108 هي مكتب اشتراكات أتوبيسات جامعة MSA الرسمي لإدارة خطوط ومواعيد النقل الجامعي.",
    ),
    "K110": (
        "Room K110 is a lecture and workshop room in Building K used for Faculty of Computer Science and Pharmacy sessions.",
        "قاعة K110 هي قاعة محاضرات وورش عمل في مبنى K تُستخدم لكلية علوم الحاسب وكلية الصيدلة.",
    ),
    "K312": (
        "Room K312 is a lab and seminar room on the Third floor of Building K used for Computer Science and technical sessions.",
        "قاعة K312 هي معمل وقاعة دراسية في الدور الثالث بمبنى K لكلية علوم الحاسب والأنشطة التقنية.",
    ),
    "M108": (
        "Room M108 is a clinical workshop and practical training room on the First floor of Building M for the Faculty of Physical Therapy.",
        "قاعة M108 هي قاعة تدريب إكلينيكي وورش عملية في الدور الأول بمبنى M لكلية العلاج الطبيعي.",
    ),
}

FLOOR_LABELS_EN = {
    "0": "Ground floor",
    "1": "First floor",
    "2": "Second floor",
    "3": "Third floor",
    "4": "Fourth floor",
    "5": "Fifth floor",
}

FLOOR_LABELS_AR = {
    "0": "الدور الأرضي",
    "1": "الدور الأول",
    "2": "الدور الثاني",
    "3": "الدور الثالث",
    "4": "الدور الرابع",
    "5": "الدور الخامس",
}

FACILITY_ROUTES: List[Tuple[List[str], str, str]] = [
    (
        ["library", "books", "المكتبه", "مكتبه", "كتب"],
        "The Main University Library at MSA University is located in Building 7 (alongside the MSA Gymnasium and sports facilities).",
        "المكتبة الجامعية الرئيسية في جامعة MSA موجودة في مبنى 7 بجانب جيم الجامعة والمرافق الرياضية.",
    ),
    (
        ["gym", "gymnasium", "sports", "padel", "football", "جيم", "الجيم", "ملاعب", "رياضه", "بادل"],
        "The MSA Gymnasium is located in Building 7 next to the Main Library. Outdoor sports facilities across the 50 acre campus include football pitches, basketball and volleyball courts, and a padel court.",
        "جيم جامعة MSA موجود في مبنى 7 بجوار المكتبة الرئيسية، كما يضم الحرم الجامعي ملاعب كرة قدم وسلة وطائرة وملعب بادل.",
    ),
    (
        ["bus", "buses", "transport", "transportation", "اتوبيس", "اتوبيسات", "باص", "نقل", "مواصلات"],
        "The official MSA University Bus Room is Room K108 (Building K, First floor, Room 08), where students subscribe to campus bus routes covering Greater Cairo, Giza, 6th of October, Sheikh Zayed, New Cairo, Maadi, Nasr City, Heliopolis, and Madinaty.",
        "مكتب اشتراكات أتوبيسات جامعة MSA الرسمي هو غرفة K108 في مبنى K بالدور الأول، ويغطي خطوط النقل في 6 أكتوبر والشيخ زايد والجيزة والقاهرة الكبرى.",
    ),
    (
        ["opera", "theater", "theatre", "cinema", "اوبرا", "مسرح", "سينما"],
        "The 1,150 seat MSA Opera House / Indoor Theater is located in Building G. The Main Theater and conference halls are in Building 12, and the renovated Campus Cinema Hall is inside Building 2 (SSB — Student Services Building).",
        "دار أوبرا جامعة MSA بسعة 1150 مقعداً موجودة في مبنى G، والمسرح الرئيسي وقاعات المؤتمرات في مبنى 12، وقاعة السينما الجامعية في مبنى 2 (مبنى خدمات الطلاب SSB).",
    ),
    (
        ["student affairs", "ssb", "registration office", "شؤون الطلاب", "شئون الطلاب", "خدمات الطلاب", "تسجيل"],
        "Student Affairs, registration services, student activities, and career offices are located in Building B (SSB — Student Services Building).",
        "مكاتب شؤون الطلاب والتسجيل والأنشطة الطلابية موجودة في مبنى B (مبنى خدمات الطلاب SSB).",
    ),
    (
        ["dentistry", "dental", "clinic", "clinics", "اسنان", "الاسنان", "عيادات", "عياده"],
        "Faculty of Dentistry classrooms are in Building C, and Dentistry Outpatient Clinics and labs are in Building D and Building H.",
        "تقع كلية طب الأسنان في مبنى C، وتوجد عيادات الأسنان الخارجية والمعامل في مبنى D ومبنى H.",
    ),
    (
        ["workshop", "workshops", "ورش", "ورشه"],
        "Engineering and Design technical workshops are located in Building F.",
        "تقع ورش الهندسة والتصميم الفنية في مبنى F.",
    ),
    (
        ["computer science", "cs", "علوم الحاسب", "حاسبات"],
        "The Faculty of Computer Science is housed in Building C, with additional computer labs and lecture rooms in Building K (such as K110 and K312), Building G, and Building H.",
        "تقع كلية علوم الحاسب في مبنى C، كما تستخدم معامل كمبيوتر وقاعات إضافية في مبنى K (مثل K110 وK312) ومبنى G ومبنى H.",
    ),
    (
        ["engineering", "architecture department", "هندسه", "الهندسه", "معماري", "عماره"],
        "The Faculty of Engineering (including Architectural Systems Engineering) is located in Building C, supported by technical workshops in Building F, computer labs in Building G, and the sustainable MSA Center of Earth.",
        "تقع كلية الهندسة (بما فيها قسم هندسة النظم المعمارية) في مبنى C، وتدعمها ورش الهندسة في مبنى F ومعامل مبنى G ومركز الأرض المستدام.",
    ),
    (
        ["pharmacy", "biotechnology", "biotech", "صيدله", "تكنولوجيا حيويه", "بايوتك"],
        "The Faculty of Pharmacy and Faculty of Biotechnology are housed in Building C, with specialized scientific labs in Building D and additional sessions in Building K.",
        "تقع كليتا الصيدلة والتكنولوجيا الحيوية في مبنى C، وتوجد المعامل العلمية المتخصصة في مبنى D مع قاعات إضافية في مبنى K.",
    ),
    (
        ["physical therapy", "علاج طبيعي"],
        "The Faculty of Physical Therapy utilizes specialized clinical workshops and practical training rooms primarily in Building M (such as Room M108).",
        "تستخدم كلية العلاج الطبيعي القاعات وورش التدريب الإكلينيكي الرئيسية في مبنى M (مثل قاعة M108).",
    ),
    (
        ["arts", "design", "فنون", "تصميم"],
        "The Faculty of Arts and Design (Interior Architecture, Graphics, and Media Arts) is located in Building E, with exhibition spaces in Building G and workshops in Building F.",
        "تقع كلية الفنون والتصميم في مبنى E، مع قاعات المعارض في مبنى G والورش في مبنى F.",
    ),
    (
        ["languages", "mass communication", "management", "administration", "لغات", "اعلام", "اداره", "اداريه"],
        "University Administration, Faculty of Languages, Faculty of Mass Communication, and Faculty of Management Sciences are located in Building A.",
        "تقع الإدارة الجامعية وكليات اللغات والإعلام والعلوم الإدارية في مبنى A.",
    ),
    (
        ["mosque", "مسجد", "جامع"],
        "The El Degwi Mosque is located inside the 50 acre MSA University campus along the green central walkways.",
        "يقع مسجد الدجوي داخل الحرم الجامعي لجامعة MSA على الممرات الخضراء الرئيسية.",
    ),
    (
        ["food court", "food", "restaurant", "restaurants", "cafeteria", "cafeterias", "cafe", "cafes", "coffee", "eat", "dining", "فود كورت", "كافتيريا", "كافتيريات", "مطاعم", "مطعم", "كافيه", "اكل", "قهوه"],
        "MSA University Food Court restaurants and cafes:\n• Qadra (قدرة)\n• Krispy Kreme\n• Cinnabon\n• Cilantro\n• TBS (The Bakery Shop)\n• Batates & Zalabya\n• Abu Auf\n• Costa Coffee\n• Papa John's",
        "المطاعم والكافيهات الموجودة في فود كورت جامعة MSA:\n• قدرة (Qadra)\n• كريسبي كريم (Krispy Kreme)\n• سينابون (Cinnabon)\n• سيلانترو (Cilantro)\n• تي بي إس (TBS - The Bakery Shop)\n• بطاطس وزلابيا (Batates & Zalabya)\n• أبو عوف (Abu Auf)\n• كوستا كوفي (Costa Coffee)\n• بابا جونز (Papa John's)",
    ),
]


MSA_GATE_DIRECTIONS: Dict[str, Dict[str, Dict[str, str]]] = {
    "A": {
        "1": {
            "en": "Enter through Gate 1 (nearest gate); Building A is directly ahead on your left at the main entrance plaza.",
            "ar": "ادخل من بوابة 1 (أقرب بوابة)؛ ستجد مبنى A أمامك مباشرة على اليسار عند الساحة الرئيسية.",
        },
        "2": {
            "en": "Enter through Gate 2 by the Campus Garden, turn right along the front walkway past Building B (SSB) to reach Building A.",
            "ar": "ادخل من بوابة 2 بجانب حديقة الحرم، واتجه يميناً على الممشى الأمامي بعد مبنى B (SSB) لتصل إلى مبنى A.",
        },
        "3": {
            "en": "Enter through Gate 3, walk along the main spine past Building C and Building B (SSB) toward the front plaza to reach Building A.",
            "ar": "ادخل من بوابة 3 وسر في الممشى الرئيسي مروراً بمبنى C ومبنى B (SSB) نحو الساحة الأمامية لتصل إلى مبنى A.",
        },
        "4": {
            "en": "Enter through Gate 4 near the bus zone, walk straight across the central campus walkway toward the main front plaza to reach Building A.",
            "ar": "ادخل من بوابة 4 بالقرب من منطقة الأتوبيسات، وسر للأمام عبر الممشى الرئيسي نحو الساحة الأمامية لتصل إلى مبنى A.",
        },
    },
    "B": {
        "1": {
            "en": "Enter through Gate 1, walk straight past Building A on your left to reach Building B (SSB) right ahead.",
            "ar": "ادخل من بوابة 1 وسر للأمام بعد مبنى A على يسارك لتصل مباشرة إلى مبنى B (مبنى خدمات الطلاب SSB).",
        },
        "2": {
            "en": "Enter through Gate 2 (nearest gate) beside the Campus Garden; Building B (SSB) is immediately ahead.",
            "ar": "ادخل من بوابة 2 (أقرب بوابة) بجانب حديقة الحرم الجامعي؛ ستجد مبنى B (SSB) أمامك مباشرة.",
        },
        "3": {
            "en": "Enter through Gate 3, walk straight past Building D and Building C toward the Campus Garden to reach Building B (SSB).",
            "ar": "ادخل من بوابة 3 وسر للأمام مروراً بمبنى D ومبنى C باتجاه حديقة الحرم لتصل إلى مبنى B (SSB).",
        },
        "4": {
            "en": "Enter through Gate 4, walk along the central walkway past the food court toward the Campus Garden to reach Building B (SSB).",
            "ar": "ادخل من بوابة 4 وسر في الممشى الأوسط مروراً بمنطقة المطاعم نحو حديقة الحرم لتصل إلى مبنى B (SSB).",
        },
    },
    "C": {
        "1": {
            "en": "Enter through Gate 1, walk straight past Building A and Building B (SSB) along the main walkway to reach Building C.",
            "ar": "ادخل من بوابة 1 وسر للأمام مروراً بمبنى A ومبنى B (SSB) على الممشى الرئيسي لتصل إلى مبنى C.",
        },
        "2": {
            "en": "Enter through Gate 2 (nearest gate) beside the Campus Garden, walk straight past Building B (SSB) directly into Building C.",
            "ar": "ادخل من بوابة 2 (أقرب بوابة) بجانب حديقة الحرم الجامعي، وسر للأمام مباشرة بعد مبنى B (SSB) لتدخل مبنى C.",
        },
        "3": {
            "en": "Enter through Gate 3 from the Mehwar side, walk straight past Building D into Building C.",
            "ar": "ادخل من بوابة 3 من جهة المحور، وسر للأمام بعد مبنى D لتصل مباشرة إلى مبنى C.",
        },
        "4": {
            "en": "Enter through Gate 4 near the bus zone, walk across the central walkway past the food court to reach Building C.",
            "ar": "ادخل من بوابة 4 بالقرب من منطقة الأتوبيسات، واعبر الممشى الأوسط بعد منطقة المطاعم لتصل إلى مبنى C.",
        },
    },
    "D": {
        "1": {
            "en": "Enter through Gate 1, walk past Building A, Building B (SSB), and Building C to reach Building D.",
            "ar": "ادخل من بوابة 1 وسر مروراً بمبنى A ومبنى B (SSB) ومبنى C لتصل إلى مبنى D.",
        },
        "2": {
            "en": "Enter through Gate 2 beside the Campus Garden, continue past Building C along the academic walkway to Building D.",
            "ar": "ادخل من بوابة 2 بجانب حديقة الحرم، وأكمل السير بعد مبنى C على الممشى الأكاديمي لتصل إلى مبنى D.",
        },
        "3": {
            "en": "Enter through Gate 3 (nearest gate) from the Mehwar side; Building D is right ahead on your left.",
            "ar": "ادخل من بوابة 3 (أقرب بوابة) من جهة المحور؛ ستجد مبنى D أمامك مباشرة على اليسار.",
        },
        "4": {
            "en": "Enter through Gate 4, walk along the main walkway past Building G and Building C toward the labs wing to reach Building D.",
            "ar": "ادخل من بوابة 4 وسر في الممشى الرئيسي مروراً بمبنى G ومبنى C باتجاه جناح المعامل لتصل إلى مبنى D.",
        },
    },
    "E": {
        "1": {
            "en": "Enter through Gate 1, walk straight through the central plaza past Building B and Building C toward the Arts wing to reach Building E.",
            "ar": "ادخل من بوابة 1 وسر عبر الساحة الوسطى مروراً بمبنى B ومبنى C نحو جناح الفنون لتصل إلى مبنى E.",
        },
        "2": {
            "en": "Enter through Gate 2 beside the Campus Garden, walk past Building C and bear left toward Building E.",
            "ar": "ادخل من بوابة 2 بجانب حديقة الحرم، وسر بعد مبنى C واتجه يساراً لتصل إلى مبنى E.",
        },
        "3": {
            "en": "Enter through Gate 3 (nearest gate), walk straight past the workshops and Building D to reach Building E.",
            "ar": "ادخل من بوابة 3 (أقرب بوابة) وسر للأمام بعد الورش ومبنى D لتصل إلى مبنى E.",
        },
        "4": {
            "en": "Enter through Gate 4, walk past Building G exhibition plaza and continue straight to reach Building E.",
            "ar": "ادخل من بوابة 4 وسر مروراً بساحة معارض مبنى G وأكمل للأمام لتصل إلى مبنى E.",
        },
    },
    "G": {
        "1": {
            "en": "Enter through Gate 1, walk along the main central boulevard past SSB and the food court toward the Opera House to reach Building G.",
            "ar": "ادخل من بوابة 1 وسر في الممشى الرئيسي مروراً بمبنى SSB ومنطقة المطاعم باتجاه دار الأوبرا لتصل إلى مبنى G.",
        },
        "2": {
            "en": "Enter through Gate 2 beside the Campus Garden, walk straight across the green plaza past the food court to reach Building G.",
            "ar": "ادخل من بوابة 2 بجانب حديقة الحرم، واعبر الساحة الخضراء مروراً بمنطقة المطاعم لتصل إلى مبنى G.",
        },
        "3": {
            "en": "Enter through Gate 3, walk across past Building E and the workshops toward the central auditorium zone to reach Building G.",
            "ar": "ادخل من بوابة 3 واعبر بعد مبنى E والورش باتجاه منطقة المدرجات الكبرى لتصل إلى مبنى G.",
        },
        "4": {
            "en": "Enter through Gate 4 (nearest gate) near the bus stop; Building G (Opera House landmark) is directly ahead.",
            "ar": "ادخل من بوابة 4 (أقرب بوابة) بالقرب من موقف الأتوبيسات؛ ستجد مبنى G (دار الأوبرا) أمامك مباشرة.",
        },
    },
    "H": {
        "1": {
            "en": "Enter through Gate 1, walk past Building A, Building C, and Building D toward the Research Center wing to reach Building H.",
            "ar": "ادخل من بوابة 1 وسر مروراً بمبنى A ومبنى C ومبنى D نحو جناح مركز الأبحاث لتصل إلى مبنى H.",
        },
        "2": {
            "en": "Enter through Gate 2, walk straight past Building C and Building D to reach Building H (Research Center).",
            "ar": "ادخل من بوابة 2 وسر للأمام بعد مبنى C ومبنى D لتصل إلى مبنى H (مركز الأبحاث).",
        },
        "3": {
            "en": "Enter through Gate 3 (nearest gate), walk directly toward the medical and research block beside Building D to reach Building H.",
            "ar": "ادخل من بوابة 3 (أقرب بوابة) وتوجه مباشرة إلى جناح الأبحاث والعيادات بجوار مبنى D لتصل إلى مبنى H.",
        },
        "4": {
            "en": "Enter through Gate 4, walk past Building G and Building 11 toward the medical wing to reach Building H.",
            "ar": "ادخل من بوابة 4 وسر بعد مبنى G ومبنى 11 نحو الجناح الطبي لتصل إلى مبنى H.",
        },
    },
    "K": {
        "1": {
            "en": "Enter through Gate 1, walk through the main plaza past SSB and Building C toward the bus and labs side to reach Building K.",
            "ar": "ادخل من بوابة 1 وسر عبر الساحة الرئيسية بعد مبنى SSB ومبنى C نحو جهة الأتوبيسات والمعامل لتصل إلى مبنى K.",
        },
        "2": {
            "en": "Enter through Gate 2 beside the Campus Garden, walk across the food court plaza toward Building G and Building K.",
            "ar": "ادخل من بوابة 2 بجانب حديقة الحرم، واعبر ساحة المطاعم باتجاه مبنى G لتصل إلى مبنى K.",
        },
        "3": {
            "en": "Enter through Gate 3, walk across past Building 6 Workshops and Building 7 Library toward the bus zone to reach Building K.",
            "ar": "ادخل من بوابة 3 واعبر بعد ورش مبنى 6 ومكتبة مبنى 7 باتجاه منطقة الأتوبيسات لتصل إلى مبنى K.",
        },
        "4": {
            "en": "Enter through Gate 4 (nearest gate) at the university bus zone; Building K (Bus Room K108 and labs) is right beside the entrance.",
            "ar": "ادخل من بوابة 4 (أقرب بوابة) عند منطقة أتوبيسات الجامعة؛ ستجد مبنى K (وغرفة الأتوبيسات K108) بجوار البوابة مباشرة.",
        },
    },
    "M": {
        "1": {
            "en": "Enter through Gate 1, follow the main campus walkway past SSB and the food court toward the sports and clinical zone to reach Building M.",
            "ar": "ادخل من بوابة 1 واتبع الممشى الرئيسي بعد مبنى SSB ومنطقة المطاعم نحو منطقة الملاعب والعلاج الطبيعي لتصل إلى مبنى M.",
        },
        "2": {
            "en": "Enter through Gate 2, cross the green plaza past Building 7 (Library and Gym) to reach Building M.",
            "ar": "ادخل من بوابة 2 واعبر الساحة الخضراء بعد مبنى 7 (المكتبة والجيم) لتصل إلى مبنى M.",
        },
        "3": {
            "en": "Enter through Gate 3, walk past Building E and Building 6 toward the sports courts to reach Building M.",
            "ar": "ادخل من بوابة 3 وسر بعد مبنى E ومبنى 6 باتجاه الملاعب الرياضية لتصل إلى مبنى M.",
        },
        "4": {
            "en": "Enter through Gate 4 (nearest gate), walk past Building K and the sports area directly into Building M.",
            "ar": "ادخل من بوابة 4 (أقرب بوابة) وسر بعد مبنى K والمنطقة الرياضية مباشرة إلى مبنى M.",
        },
    },
}

BUILDING_GROUP_MAP = {
    "1": "A", "A": "A",
    "2": "B", "B": "B", "SSB": "B",
    "3": "C", "C": "C",
    "4": "D", "D": "D",
    "5": "E", "E": "E",
    "6": "E", "F": "E",
    "7": "M",
    "G": "G", "12": "G",
    "H": "H", "11": "H",
    "K": "K", "L": "K",
    "M": "M", "N": "M",
}


def _extract_gate_from_question(question: str, entry_gate: str = "") -> Optional[str]:
    """Detect whether the user specified Gate 1, 2, 3, or 4 in text or via button selection."""
    match = re.search(r"(?:\bgate\s*|بوابة\s*|بوابه\s*|باب\s*)([1-4])\b", question, re.IGNORECASE)
    if match:
        return match.group(1)
    stripped = question.strip()
    if stripped in {"1", "2", "3", "4"}:
        return stripped
    cleaned_gate = str(entry_gate or "").strip()
    if cleaned_gate in {"1", "2", "3", "4"}:
        return cleaned_gate
    return None


def _build_gate_options(target_code: str, is_ar: bool) -> List[Dict[str, str]]:
    """Return clickable Gate 1-4 option payloads for the Avatar chat bubble."""
    return [
        {
            "gate": g,
            "label": f"بوابة {g}" if is_ar else f"Gate {g}",
            "target": target_code,
            "query": f"بوابة {g}" if is_ar else f"Gate {g}",
        }
        for g in ("1", "2", "3", "4")
    ]


def _build_single_gate_direction(bldg_key: str, floor_step_en: str, floor_step_ar: str, gate: str, is_ar: bool) -> str:
    """Format concise walking directions from a single chosen gate."""
    group = BUILDING_GROUP_MAP.get(bldg_key, "C")
    routes = MSA_GATE_DIRECTIONS.get(group, MSA_GATE_DIRECTIONS["C"])
    step = routes[gate]["ar" if is_ar else "en"]
    if is_ar:
        return f"الاتجاهات من بوابة {gate}: {step}{floor_step_ar}"
    return f"Directions from Gate {gate}: {step}{floor_step_en}"


def resolve_msa_room_or_location(
    question: str,
    chunks: List[Dict[str, Any]],
    entry_gate: str = "",
    pending_target: str = "",
) -> Optional[Dict[str, Any]]:
    """Decode MSA room/building codes. If no gate is given, ask which gate (1-4) with interactive options."""
    q_clean = clean_text(question)
    q_lang = detect_language(q_clean)
    is_ar = q_lang == "ar"
    chosen_gate = _extract_gate_from_question(q_clean, entry_gate=entry_gate)

    # If the user replied with just a gate ("Gate 2", "2", "بوابة 3") after a room question, resolve using pending_target
    effective_query = q_clean
    is_gate_followup = False
    if pending_target and chosen_gate and not re.search(r"\b(SSB|[ABCDEFGHKLMN])\s*[-#]?\s*([0-5]\d{2})\b", q_clean, re.IGNORECASE):
        if not re.search(r"(?:\bbuilding\s+|مبنى\s+|مبني\s+)(SSB|1[0-3]|[1-9]|[ABCDEFGHKLMN])\b", q_clean, re.IGNORECASE):
            effective_query = f"{pending_target} Gate {chosen_gate}"
            is_gate_followup = True

    # 1. Exact room code detection like C211, C111, K108, G204, B005, M108, A104, D301, SSB 201
    room_match = re.search(r"\b(SSB|[ABCDEFGHKLMN])\s*[-#]?\s*([0-5]\d{2})\b", effective_query, re.IGNORECASE)
    if room_match:
        bldg_key = room_match.group(1).upper()
        room_num = room_match.group(2)
        full_code = f"{bldg_key}{room_num}"
        floor_digit = room_num[0]
        unit_num = room_num[1:]
        bldg = MSA_BUILDINGS.get(bldg_key)

        uploaded_notes = [
            c for c in chunks if re.search(rf"\b{re.escape(full_code)}\b", c.get("text", ""), re.IGNORECASE)
        ]

        if bldg:
            is_ground_level = floor_digit in {"0", "1"}
            if is_ar:
                floor_label = FLOOR_LABELS_AR.get(floor_digit, f"الدور {floor_digit}")
                header = f"قاعة {full_code} موجودة في {bldg['ar_name']}، {floor_label}، غرفة {unit_num}."
                if is_ground_level:
                    floor_step_ar = f" تقع غرفة {unit_num} مباشرة على مستوى الأرض ({floor_label}) بدون الحاجة لسلم أو مصعد."
                else:
                    floor_step_ar = f" ثم اصعد بالسلم أو المصعد إلى {floor_label}، غرفة {unit_num}."
                floor_step_en = ""
            else:
                floor_label = FLOOR_LABELS_EN.get(floor_digit, f"Floor {floor_digit}")
                header = f"Room {full_code} is located in {bldg['en_name']}, {floor_label}, Room {unit_num}."
                if is_ground_level:
                    floor_step_en = f" Room {unit_num} is right on the ground level ({floor_label}), no stairs or elevator needed."
                else:
                    floor_step_en = f" Then take the stairs or elevator to the {floor_label}, Room {unit_num}."
                floor_step_ar = ""

            matches = [
                {
                    "id": f"msa_room_{full_code.lower()}",
                    "kind": "room_locator",
                    "title": f"MSA Room {full_code} · {bldg['en_name']}",
                    "excerpt": header,
                    "source_file": "MSA_Rooms_and_Locations_Guide.md",
                    "course": "MSA Campus Map & Gate Directions",
                    "confidence": 0.99,
                }
            ]
            for extra in uploaded_notes[:2]:
                if extra["source_file"] != "MSA_Rooms_and_Locations_Guide.md":
                    matches.append(
                        {
                            "id": extra["id"],
                            "kind": "chunk",
                            "title": extra["heading"],
                            "excerpt": extra["text"][:260],
                            "source_file": extra["source_file"],
                            "course": extra["course"],
                            "confidence": 0.92,
                        }
                    )

            # Two-step interactive flow: if gate not yet chosen, ask which gate they entered from!
            if not chosen_gate:
                ask_gate = (
                    "من أي بوابة دخلت الجامعة؟ اختر بوابة 1 أو بوابة 2 أو بوابة 3 أو بوابة 4 لأعطيك الاتجاهات."
                    if is_ar
                    else "Which gate did you enter from? Choose Gate 1, Gate 2, Gate 3, or Gate 4 for directions."
                )
                return {
                    "answer": f"{header}\n\n{ask_gate}",
                    "matches": matches,
                    "mode": "msa_room_locator",
                    "needs_gate": True,
                    "pending_target": full_code,
                    "gate_options": _build_gate_options(full_code, is_ar),
                }

            direction = _build_single_gate_direction(bldg_key, floor_step_en, floor_step_ar, chosen_gate, is_ar)
            answer_text = direction if is_gate_followup else f"{header}\n\n{direction}"
            return {
                "answer": answer_text,
                "matches": matches,
                "mode": "msa_room_locator",
                "needs_gate": False,
                "pending_target": "",
                "gate_options": [],
            }

    # 2. Specific building lookup (e.g. "Where is Building G?", "فين مبنى SSB؟")
    bldg_match = re.search(
        r"(?:\bbuilding\s+|مبنى\s+|مبني\s+)(SSB|1[0-3]|[1-9]|[ABCDEFGHKLMN])\b|\b(SSB)\b",
        effective_query,
        re.IGNORECASE,
    )
    is_multi_bldg = len(re.findall(r"(?:\bbuilding\s+|مبنى\s+|مبني\s+)(?:SSB|1[0-3]|[1-9]|[ABCDEFGHKLMN])\b", effective_query, re.IGNORECASE)) > 1
    is_directory_q = bool(re.search(r"\b(directory|all buildings|buildings list)\b|(دليل مباني|كل المباني|مباني جامعة)", effective_query, re.IGNORECASE))
    if bldg_match and not is_multi_bldg and not is_directory_q:
        code = (bldg_match.group(1) or bldg_match.group(2) or "").upper()
        bldg = MSA_BUILDINGS.get(code)
        if bldg:
            target_label = f"Building {code}" if code != "SSB" else "SSB"
            header = f"{bldg['ar_name']}." if is_ar else f"{bldg['en_name']}."
            matches = [
                {
                    "id": f"msa_bldg_{code.lower()}",
                    "kind": "room_locator",
                    "title": bldg["ar_name"] if is_ar else bldg["en_name"],
                    "excerpt": header,
                    "source_file": "MSA_Rooms_and_Locations_Guide.md",
                    "course": "MSA Campus Map & Gate Directions",
                    "confidence": 0.98,
                }
            ]
            if not chosen_gate:
                ask_gate = (
                    "من أي بوابة دخلت الجامعة؟ اختر بوابة 1 أو بوابة 2 أو بوابة 3 أو بوابة 4 لأعطيك الاتجاهات."
                    if is_ar
                    else "Which gate did you enter from? Choose Gate 1, Gate 2, Gate 3, or Gate 4 for directions."
                )
                return {
                    "answer": f"{header}\n\n{ask_gate}",
                    "matches": matches,
                    "mode": "msa_room_locator",
                    "needs_gate": True,
                    "pending_target": target_label,
                    "gate_options": _build_gate_options(target_label, is_ar),
                }

            direction = _build_single_gate_direction(code, "", "", chosen_gate, is_ar)
            return {
                "answer": direction if is_gate_followup else f"{header}\n\n{direction}",
                "matches": matches,
                "mode": "msa_room_locator",
                "needs_gate": False,
                "pending_target": "",
                "gate_options": [],
            }

    # 3. Direct "Where is <facility/faculty>?" or food court / dining intent
    location_intent = bool(
        re.search(
            r"\b(where|location|located|find|room|hall|floor|building|map|directions|food court|restaurant|restaurants|cafeteria|cafeterias|eat|dining)\b|(فين|أين|اين|مكان|موقع|قاعة|قاعه|غرفة|غرفه|مبنى|مبني|مدرج|فود كورت|مطاعم|مطعم|كافتيريا|اكل)",
            q_clean,
            re.IGNORECASE,
        )
    )
    is_fees_q = bool(
        re.search(
            r"\b(fee|fees|tuition|cost|price|how much|payment|refund|scholarship|discount)\b|(مصاريف|مصروفات|رسوم|سعر|بكام|كام|فلوس|دفع|سداد|منح|خصم)",
            q_clean,
            re.IGNORECASE,
        )
    )
    if location_intent and not is_directory_q and not is_fees_q:
        q_norm_tokens = set(_tokenize(q_clean))
        q_norm_str = " ".join(_tokenize(q_clean))
        for keywords, ans_en, ans_ar in FACILITY_ROUTES:
            matched_kw = False
            for kw in keywords:
                if re.search(r"[\u0600-\u06FF]", kw):
                    kw_norm = " ".join(_tokenize(kw))
                    if " " in kw_norm:
                        matched_kw = kw_norm in q_norm_str
                    else:
                        matched_kw = kw_norm in q_norm_tokens
                else:
                    matched_kw = bool(re.search(rf"\b{re.escape(kw)}\b", q_clean, re.IGNORECASE))
                if matched_kw:
                    break
            if matched_kw:
                answer = ans_ar if is_ar else ans_en
                return {
                    "answer": answer,
                    "matches": [
                        {
                            "id": "msa_facility_match",
                            "kind": "room_locator",
                            "title": "MSA Campus Facility Directory",
                            "excerpt": answer[:260],
                            "source_file": "MSA_Rooms_and_Locations_Guide.md",
                            "course": "MSA Campus Map & Gate Directions",
                            "confidence": 0.96,
                        }
                    ],
                    "mode": "msa_room_locator",
                    "needs_gate": False,
                    "pending_target": "",
                    "gate_options": [],
                }
    return None


def _fuzzy_contains(tokens: List[str], targets: List[str], threshold: float = 0.76) -> bool:
    """Typo-tolerant token matcher (handles 'fucalties', 'engineerin', 'univeersty', etc.)."""
    for tok in tokens:
        if len(tok) < 4:
            continue
        for target in targets:
            if tok == target or SequenceMatcher(None, tok, target).ratio() >= threshold:
                return True
    return False


def _match_specific_faculty(question: str) -> Optional[str]:
    """Match a specific MSA faculty name, abbreviation, or typo from user input."""
    q_low = clean_text(question).lower()
    q_tokens = _tokenize(question)
    q_norm_str = " ".join(q_tokens)

    for fac_key, info in MSA_FACULTIES_DATA.items():
        for alias in info["aliases"]:
            if re.search(r"[\u0600-\u06FF]", alias):
                alias_norm = " ".join(_tokenize(alias))
                if (" " in alias_norm and alias_norm in q_norm_str) or (alias_norm in q_tokens):
                    return fac_key
            elif re.search(rf"\b{re.escape(alias)}\b", q_low):
                return fac_key

    for fac_key, info in MSA_FACULTIES_DATA.items():
        en_aliases = [a for a in info["aliases"] if not re.search(r"[\u0600-\u06FF]", a) and " " not in a and len(a) >= 5]
        if _fuzzy_contains(q_tokens, en_aliases, threshold=0.78):
            return fac_key
    return None


def _match_specific_major(question: str, restrict_faculty: Optional[str] = None) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Match a specific major inside a given faculty (or across all 11 faculties)."""
    q_low = clean_text(question).lower()
    q_tokens = _tokenize(question)
    q_norm_str = " ".join(q_tokens)

    fac_keys = [restrict_faculty] if restrict_faculty and restrict_faculty in MSA_FACULTIES_DATA else list(MSA_FACULTIES_DATA.keys())

    # 1. Exact full major name match (from button click)
    for fk in fac_keys:
        for major in MSA_FACULTIES_DATA[fk]["majors"]:
            if q_low == major["name_en"].lower() or clean_text(question) == major["name_ar"]:
                return fk, major

    # 2. Alias / keyword match
    for fk in fac_keys:
        for major in MSA_FACULTIES_DATA[fk]["majors"]:
            for alias in major["aliases"]:
                if re.search(r"[\u0600-\u06FF]", alias):
                    alias_norm = " ".join(_tokenize(alias))
                    if (" " in alias_norm and alias_norm in q_norm_str) or (alias_norm in q_tokens):
                        return fk, major
                elif re.search(rf"\b{re.escape(alias)}\b", q_low):
                    return fk, major

    # 3. Fuzzy typo match on single-word English aliases
    for fk in fac_keys:
        for major in MSA_FACULTIES_DATA[fk]["majors"]:
            en_aliases = [a for a in major["aliases"] if not re.search(r"[\u0600-\u06FF]", a) and " " not in a and len(a) >= 5]
            if _fuzzy_contains(q_tokens, en_aliases, threshold=0.78):
                return fk, major

    return None


def _format_faculty_majors_response(fac_key: str, is_ar: bool) -> Dict[str, Any]:
    """Step 2: Bullet points of the faculty's majors + 'Which major are you interested in?' + clickable major buttons."""
    fac = MSA_FACULTIES_DATA[fac_key]
    majors = fac["majors"]

    if is_ar:
        header = f"كلية {fac['label_ar']} ({fac['duration_ar']} · {fac['partner_ar']}):"
        bullets = "\n".join(f"• {m['name_ar']}" for m in majors)
        prompt_q = "ما هو التخصص الذي تهتم به لعرض خطة المواد الكاملة؟"
    else:
        header = f"Faculty of {fac['label_en']} ({fac['duration_en']} · {fac['partner_en']}):"
        bullets = "\n".join(f"• {m['name_en']}" for m in majors)
        prompt_q = "Which major are you interested in?"

    answer = f"{header}\n{bullets}\n\n{prompt_q}"
    options = [
        {
            "gate": "",
            "label": m["name_ar"] if is_ar else m["name_en"],
            "target": f"MSA_MAJOR_SELECT:{fac_key}",
            "query": m["name_ar"] if is_ar else m["name_en"],
        }
        for m in majors
    ]
    return {
        "answer": answer,
        "matches": [
            {
                "id": f"msa_faculty_{fac_key}",
                "kind": "faculty_guide",
                "title": f"كلية {fac['label_ar']}" if is_ar else f"Faculty of {fac['label_en']}",
                "excerpt": header,
                "source_file": "MSA_Study_Plans_All_Faculties.md",
                "course": "MSA University Study Plans",
                "confidence": 0.99,
            }
        ],
        "mode": "msa_faculty_guide",
        "needs_gate": True,
        "pending_target": f"MSA_MAJOR_SELECT:{fac_key}",
        "gate_options": options,
    }


def _format_major_study_plan_response(fac_key: str, major: Dict[str, Any], is_ar: bool) -> Dict[str, Any]:
    """Step 3: Full 5-year or 4-year study plan bullets + Top 5 Frequently Asked Internet Questions."""
    title = major["plan_title_ar"] if is_ar else major["plan_title_en"]
    years = major["years_ar"] if is_ar else major["years_en"]
    bullets = "\n".join(f"• {yr}" for yr in years)

    if is_ar:
        faq_header = "أكثر 5 أسئلة شائعة عن جامعة MSA على الإنترنت (اضغط على أي سؤال للإجابة):"
        faq_lines = "\n".join(f"{idx}. {item['q_ar']}" for idx, item in enumerate(TOP_5_MSA_FAQS, start=1))
    else:
        faq_header = "Top 5 frequently asked questions about MSA University on the internet (tap any question to answer):"
        faq_lines = "\n".join(f"{idx}. {item['q_en']}" for idx, item in enumerate(TOP_5_MSA_FAQS, start=1))

    faq_options = [
        {
            "gate": "",
            "label": f"Q{idx}: {item['q_ar'] if is_ar else item['q_en']}",
            "target": f"MSA_TOP_FAQ:{fac_key}",
            "query": item["q_ar"] if is_ar else item["q_en"],
        }
        for idx, item in enumerate(TOP_5_MSA_FAQS, start=1)
    ]

    answer = f"{title}:\n\n{bullets}\n\n{faq_header}\n{faq_lines}"
    return {
        "answer": answer,
        "matches": [
            {
                "id": f"msa_plan_{fac_key}_{major['key']}",
                "kind": "study_plan",
                "title": title,
                "excerpt": years[0][:220],
                "source_file": "MSA_Study_Plans_All_Faculties.md",
                "course": f"MSA {MSA_FACULTIES_DATA[fac_key]['label_en']} Study Plan",
                "confidence": 0.99,
            }
        ],
        "mode": "msa_study_plan_rag",
        "needs_gate": False,
        "pending_target": f"MSA_TOP_FAQ:{fac_key}",
        "gate_options": faq_options,
    }


def resolve_msa_faculty_faq(question: str, pending_target: str = "") -> Optional[Dict[str, Any]]:
    """Answer Q1–Q5 specifically tailored to the faculty the student selected in the study-plan flow."""
    if not pending_target.startswith("MSA_TOP_FAQ:"):
        return None
    fac_key = pending_target.split(":", 1)[1].strip()
    if fac_key not in MSA_FACULTIES_DATA or fac_key not in MSA_TUITION_FEES:
        return None

    q_clean = clean_text(question)
    q_low = q_clean.lower()
    is_ar = detect_language(q_clean) == "ar"
    fee = MSA_TUITION_FEES[fac_key]
    fac = MSA_FACULTIES_DATA[fac_key]
    extra_meta = MSA_FACULTY_SCHOLARSHIP_PARENTS.get(fac_key, {})

    # Q4: Scholarships & Tuition Discounts tailored to the chosen faculty
    is_q4 = bool(
        re.match(r"^\s*q\s*4\b", q_low)
        or re.search(r"\b(scholarship|scholarships|discount|discounts)\b|(منح|منحة|منحه|خصم|خصومات)", q_low, re.IGNORECASE)
    )
    if is_q4:
        egp = fee["egyptian_egp"]
        uk_en = f" + {fee['uk_fee']} UK validation ({fee['uk_partner_en']})" if fee["uk_fee"] else f" ({fee['uk_partner_en']})"
        uk_ar = f" + {fee['uk_fee']} اعتماد بريطاني ({fee['uk_partner_ar']})" if fee["uk_fee"] else f" ({fee['uk_partner_ar']})"
        if is_ar:
            answer = (
                f"المنح الدراسية والخصومات الرسمية الخاصة بـ {fee['label_ar']} بجامعة MSA:\n\n"
                f"• منحة التخصص المماثل لأحد الوالدين (خصم 15% حتى التخرج بدون شرط GPA): متاحة لأبناء خريجي الكليات التالية من أي جامعة مصرية أو دولية معتمدة:\n"
                f"  {extra_meta.get('parents_ar', '')}.\n"
                f"• منحة أبناء خريجي MSA (Legacy Program): خصم 15% حتى التخرج (لا تُجمع مع خصم التخصص المماثل).\n"
                f"• منحة خريجي مدارس دار التربية و IKC: خصم 10% حتى التخرج.\n"
                f"• المنحة الرياضية لأبطال الجمهورية والمنتخبات: خصم 10% في الفصل الأول وتستمر عند تمثيل الجامعة.\n"
                f"• خصم السداد السنوي المبكر الكامل: 5% عند سداد المصروفات السنوية كاملة قبل 31 أغسطس 2026.\n"
                f"• منح التفوق الأكاديمي (تصل إلى 20% حسب المعدل التراكمي) ومنحة أوائل الثانوية العامة.\n\n"
                f"ملاحظة: تُحسب المنح الخاصة من فئة المصروفات الخاصة بالطالب في {fee['label_ar']} (2026/2027):\n"
                f"• فئة A: {egp['A']['total']} | فئة B: {egp['B']['total']} | فئة C: {egp['C']['total']} | فئة D: {egp['D']['total']}{uk_ar}"
            )
        else:
            answer = (
                f"Official Scholarships & Tuition Discounts for {fee['label_en']} (MSA University):\n\n"
                f"• 15% Similar Education Parents Scholarship (Till Graduation, not conditional on GPA): Available if a parent graduated from any accredited Egyptian or international university in:\n"
                f"  {extra_meta.get('parents_en', '')}.\n"
                f"• 15% MSA Alumni Legacy Scholarship (Till Graduation): For sons & daughters of MSA graduates (cannot be combined with Similar Education Parents discount).\n"
                f"• 10% Dar El Tarbiah & IKC Schools Scholarship (Till Graduation).\n"
                f"• 10% National Sports Champions Scholarship (First Semester, renewable when representing MSA).\n"
                f"• 5% Early Full Annual Payment Scholarship (paying full year tuition by August 31, 2026).\n"
                f"• Academic Merit Discounts (up to 20% for high CGPA) and up to 100% Top Thanaweya Amma waiver.\n\n"
                f"Note: Special scholarships can be combined and are calculated from your assigned {fee['label_en']} fee category (2026/2027 EGP):\n"
                f"• Cat A: {egp['A']['total']} | Cat B: {egp['B']['total']} | Cat C: {egp['C']['total']} | Cat D: {egp['D']['total']}{uk_en}"
            )
        return {
            "answer": answer,
            "matches": [
                {
                    "id": f"msa_scholarships_{fac_key}",
                    "kind": "faculty_faq",
                    "title": f"{fee['label_en']} Scholarships & Discounts",
                    "excerpt": answer[:260],
                    "source_file": "MSA_Tuition_Fees_and_Scholarships_Official.md",
                    "course": f"MSA {fee['label_en']} Scholarships",
                    "confidence": 0.99,
                }
            ],
            "mode": "msa_faculty_faq",
            "needs_gate": False,
            "pending_target": f"MSA_TOP_FAQ:{fac_key}",
            "gate_options": [],
        }

    # Q3: Tuition Fees tailored to the chosen faculty
    is_q3 = bool(
        re.match(r"^\s*q\s*3\b", q_low)
        or re.search(r"\b(tuition|fees|fee|annual increase|validation fee)\b|(مصروفات|مصاريف|رسوم الاعتماد)", q_low, re.IGNORECASE)
    )
    if is_q3:
        res = _format_single_faculty_tuition(fac_key, is_ar)
        res["pending_target"] = f"MSA_TOP_FAQ:{fac_key}"
        return res

    # Q1: British Dual Degree tailored to the chosen faculty
    is_q1 = bool(
        re.match(r"^\s*q\s*1\b", q_low)
        or re.search(r"\b(dual degree|equivalency|british)\b|(الشهادة البريطانية|المزدوجة|معادلة)", q_low, re.IGNORECASE)
    )
    if is_q1:
        if is_ar:
            answer = (
                f"الاعتماد الأكاديمي والشهادة في {fee['label_ar']}:\n"
                f"• {fac['partner_ar']}.\n"
                + (
                    f"• رسوم الاعتماد البريطاني السنوية: {fee['uk_fee']} تُدفع في الفصل الدراسي الأول، ويحصل الخريج على شهادتين معتمدتين (مصرية وبريطانية من {fee['uk_partner_ar']}) مما يتيح استكمال الدراسات العليا في بريطانيا بدون معادلة تقليدية."
                    if fee["uk_fee"]
                    else "• تمنح الكلية بكالوريوس طب وجراحة الفم والأسنان معتمداً بالكامل من وزارة التعليم العالي والمجلس الأعلى للجامعات المصري مع سنة امتياز وتدريب إكلينيكي مكثف."
                )
            )
        else:
            answer = (
                f"Degree Validation & Accreditation for {fee['label_en']}:\n"
                f"• {fac['partner_en']}.\n"
                + (
                    f"• Annual British validation fee: {fee['uk_fee']} (paid in 1st semester). Graduates earn two accredited degrees (Egyptian Ministry of Higher Education + {fee['uk_partner_en']}, UK) and can pursue UK postgraduate studies without traditional equivalency."
                    if fee["uk_fee"]
                    else "• Graduates earn the fully accredited Egyptian Ministry of Higher Education & Supreme Council of Universities BDS degree with full clinical training and internship."
                )
            )
        return {
            "answer": answer,
            "matches": [
                {
                    "id": f"msa_dual_{fac_key}",
                    "kind": "faculty_faq",
                    "title": f"{fee['label_en']} Degree Accreditation",
                    "excerpt": answer[:260],
                    "source_file": "MSA_Top_5_Internet_FAQs.md",
                    "course": f"MSA {fee['label_en']}",
                    "confidence": 0.99,
                }
            ],
            "mode": "msa_faculty_faq",
            "needs_gate": False,
            "pending_target": f"MSA_TOP_FAQ:{fac_key}",
            "gate_options": [],
        }

    # Q2: Egyptian Professional Syndicate tailored to the chosen faculty
    is_q2 = bool(
        re.match(r"^\s*q\s*2\b", q_low)
        or re.search(r"\b(syndicate|syndicates)\b|(النقابات|نقابة)", q_low, re.IGNORECASE)
    )
    if is_q2:
        if is_ar:
            answer = f"نعم، خريجو {fee['label_ar']} بجامعة MSA معتمدون رسمياً من وزارة التعليم العالي والمجلس الأعلى للجامعات المصري ومؤهلون للانضمام إلى: {extra_meta.get('syndicate_ar', 'النقابة المهنية المختصة')}."
        else:
            answer = f"Yes. Graduates of {fee['label_en']} at MSA University hold fully accredited Supreme Council of Universities degrees and are eligible to join the {extra_meta.get('syndicate_en', 'relevant Egyptian Professional Syndicate')}."
        return {
            "answer": answer,
            "matches": [
                {
                    "id": f"msa_syndicate_{fac_key}",
                    "kind": "faculty_faq",
                    "title": f"{fee['label_en']} Syndicate Eligibility",
                    "excerpt": answer[:260],
                    "source_file": "MSA_Top_5_Internet_FAQs.md",
                    "course": f"MSA {fee['label_en']}",
                    "confidence": 0.99,
                }
            ],
            "mode": "msa_faculty_faq",
            "needs_gate": False,
            "pending_target": f"MSA_TOP_FAQ:{fac_key}",
            "gate_options": [],
        }

    # Q5: UK Summer Abroad & Transfer GPA tailored to the chosen faculty
    is_q5 = bool(
        re.match(r"^\s*q\s*5\b", q_low)
        or re.search(r"\b(summer abroad|transfer)\b|(الصيفي|التحويل)", q_low, re.IGNORECASE)
    )
    if is_q5:
        gpa_req = extra_meta.get("transfer_gpa", "2.0")
        if is_ar:
            answer = (
                f"برنامج الدراسة الصيفي في بريطانيا والتحويل إلى {fee['label_ar']}:\n"
                f"• يمكن لطلاب {fee['label_ar']} المتميزين التقديم في برنامج الدراسة الصيفي في بريطانيا (UK Summer Abroad Programme) بالتعاون مع جامعتي جرينتش وبيدفوردشير وفقاً للمعدل التراكمي والمقابلة.\n"
                f"• شروط التحويل من جامعة أخرى إلى {fee['label_ar']}: يشترط معدل تراكمي لا يقل عن {gpa_req}، وتقديم بيان درجات ومحتوى علمي مختوم وخطاب حسن سير وسلوك، مع عدم سحب الملف من الجامعة السابقة إلا بعد الموافقة النهائية من مكتب القبول بجامعة MSA."
            )
        else:
            answer = (
                f"UK Summer Abroad & Transfer Rules for {fee['label_en']}:\n"
                f"• Eligible {fee['label_en']} students can apply to MSA's UK Summer Abroad Programme with the University of Greenwich and University of Bedfordshire based on CGPA and interview.\n"
                f"• Transferring into {fee['label_en']} requires a minimum cumulative GPA of {gpa_req}, official transcript with stamped course descriptions, Good Conduct letter, ELPT, and Faculty Interview. Do NOT withdraw from your current university until receiving explicit final approval from an MSA Admission Officer."
            )
        return {
            "answer": answer,
            "matches": [
                {
                    "id": f"msa_transfer_{fac_key}",
                    "kind": "faculty_faq",
                    "title": f"{fee['label_en']} Summer Abroad & Transfer",
                    "excerpt": answer[:260],
                    "source_file": "MSA_Admission_Guide_Official.md",
                    "course": f"MSA {fee['label_en']}",
                    "confidence": 0.99,
                }
            ],
            "mode": "msa_faculty_faq",
            "needs_gate": False,
            "pending_target": f"MSA_TOP_FAQ:{fac_key}",
            "gate_options": [],
        }

    return None


def _format_single_faculty_tuition(fac_key: str, is_ar: bool) -> Dict[str, Any]:
    """Return exact 2026/2027 Categories A, B, C, D (1st Sem, 2nd Sem, Total) in EGP & USD for one faculty."""
    fee = MSA_TUITION_FEES[fac_key]
    egp = fee["egyptian_egp"]
    usd = fee["international_usd"]

    if is_ar:
        uk_line = (
            f"• رسوم الاعتماد البريطاني ({fee['uk_partner_ar']}): + {fee['uk_fee']} سنوياً (تُدفع في الفصل الدراسي الأول)."
            if fee["uk_fee"]
            else f"• الاعتماد: {fee['uk_partner_ar']}."
        )
        egp_lines = "\n".join(
            f"• فئة {cat}: ترم أول {egp[cat]['sem1']} | ترم ثاني {egp[cat]['sem2']} | الإجمالي السنوي: {egp[cat]['total']}"
            for cat in ("A", "B", "C", "D")
        )
        usd_lines = "\n".join(
            f"• فئة {cat}: ترم أول {usd[cat]['sem1']} | ترم ثاني {usd[cat]['sem2']} | الإجمالي السنوي: {usd[cat]['total']}"
            for cat in ("A", "B", "C", "D")
        )
        intl_uk = f" (+ {fee['intl_uk_fee']} اعتماد بريطاني)" if fee["intl_uk_fee"] else ""
        extra = f"\n• ملاحظة: {fee['extra_ar']}" if fee["extra_ar"] else ""
        answer = (
            f"مصروفات {fee['label_ar']} الرسمية بجامعة MSA (2026/2027):\n\n"
            f"الطلاب المصريون (بالجنيه المصري):\n{egp_lines}\n{uk_line}{extra}\n\n"
            f"الطلاب الأجانب / غير المصريين (بالدولار الأمريكي{intl_uk}):\n{usd_lines}"
        )
    else:
        uk_line = (
            f"• British Validation Fee ({fee['uk_partner_en']}): + {fee['uk_fee']} per year (paid in 1st semester)."
            if fee["uk_fee"]
            else f"• Accreditation: {fee['uk_partner_en']}."
        )
        egp_lines = "\n".join(
            f"• Category {cat}: 1st Sem {egp[cat]['sem1']} | 2nd Sem {egp[cat]['sem2']} | Annual Total: {egp[cat]['total']}"
            for cat in ("A", "B", "C", "D")
        )
        usd_lines = "\n".join(
            f"• Category {cat}: 1st Sem {usd[cat]['sem1']} | 2nd Sem {usd[cat]['sem2']} | Annual Total: {usd[cat]['total']}"
            for cat in ("A", "B", "C", "D")
        )
        intl_uk = f" (+ {fee['intl_uk_fee']} UK fee)" if fee["intl_uk_fee"] else ""
        extra = f"\n• Note: {fee['extra_en']}" if fee["extra_en"] else ""
        answer = (
            f"Official 2026/2027 Tuition Fees for {fee['label_en']} (MSA University):\n\n"
            f"Egyptian Students (EGP):\n{egp_lines}\n{uk_line}{extra}\n\n"
            f"International / Non-Egyptian Students (USD{intl_uk}):\n{usd_lines}"
        )

    return {
        "answer": answer,
        "matches": [
            {
                "id": f"msa_tuition_{fac_key}",
                "kind": "tuition_table",
                "title": f"{fee['label_en']} Tuition Fees (2026/2027)",
                "excerpt": answer[:260],
                "source_file": "MSA_Tuition_Fees_and_Scholarships_Official.md",
                "course": "MSA Official Tuition Fees (2026/2027)",
                "confidence": 0.99,
            }
        ],
        "mode": "msa_tuition_guide",
        "needs_gate": False,
        "pending_target": "",
        "gate_options": [],
    }


def resolve_msa_tuition_fees(question: str, pending_target: str = "") -> Optional[Dict[str, Any]]:
    """Deterministic bilingual MSA 2026/2027 tuition fees table resolver (all 11 faculties x Categories A/B/C/D)."""
    q_clean = clean_text(question)
    q_low = q_clean.lower()
    is_ar = detect_language(q_clean) == "ar"

    # Follow-up click after viewing all-faculties tuition summary
    if pending_target == "MSA_TUITION_FACULTY_SELECT":
        matched_fac = _match_specific_faculty(q_clean)
        if matched_fac and matched_fac in MSA_TUITION_FEES:
            return _format_single_faculty_tuition(matched_fac, is_ar)

    # Exclude bus fees, minimum scores, refund-only, or pure scholarship questions (let RAG answer those directly)
    is_bus_q = bool(re.search(r"\b(bus|buses|transportation)\b|(اتوبيس|أتوبيس|باص|مواصلات)", q_low, re.IGNORECASE))
    is_min_score_q = bool(re.search(r"\b(minimum score|minimum scores|min score|acceptance rate|thanaweya score)\b|(الحد الادنى|الحد الأدنى|تنسيق)", q_low, re.IGNORECASE))
    is_pure_scholarship_q = bool(
        re.search(r"\b(scholarship|scholarships|discount|discounts)\b|(منح|منحة|منحه|خصم|خصومات)", q_low, re.IGNORECASE)
    ) and not bool(
        re.search(r"\b(tuition fees|tuition fee|category|categories|price|prices|how much|cost)\b|(مصاريف|مصروفات|فئات|بكام|سعر)", q_low, re.IGNORECASE)
    )
    if is_bus_q or is_min_score_q or is_pure_scholarship_q:
        return None

    asks_tuition = bool(
        re.search(
            r"\b(tuition|fees|fee|category|categories|price|prices|cost|how much)\b|(مصاريف|مصروفات|المصاريف|المصروفات|رسوم|فئات|بكام|كام مصاريف|سعر)",
            q_low,
            re.IGNORECASE,
        )
    )
    if not asks_tuition:
        return None

    # Ignore unrelated course/general price questions (e.g. "price of bitcoin") unless MSA/faculty/tuition context is present
    specific_fac = _match_specific_faculty(q_clean)
    has_msa_or_tuition_ctx = bool(
        specific_fac
        or re.search(
            r"\b(tuition|msa|faculty|faculties|category|categories|egyptian|international|semester|british validation|university)\b|(مصاريف|مصروفات|المصاريف|المصروفات|رسوم|فئات|كلية|كليات|جامعة|msa)",
            q_low,
            re.IGNORECASE,
        )
    )
    if not has_msa_or_tuition_ctx:
        return None

    if specific_fac and specific_fac in MSA_TUITION_FEES:
        return _format_single_faculty_tuition(specific_fac, is_ar)

    # All 11 faculties complete 2026/2027 Categories A / B / C / D summary + 11 clickable buttons for full semester & USD details
    lines: List[str] = []
    for fk, fee in MSA_TUITION_FEES.items():
        egp = fee["egyptian_egp"]
        uk = f" + {fee['uk_fee']}" if fee["uk_fee"] else ""
        if is_ar:
            lines.append(
                f"• {fee['label_ar']}: A: {egp['A']['total']} | B: {egp['B']['total']} | C: {egp['C']['total']} | D: {egp['D']['total']}{uk}"
            )
        else:
            lines.append(
                f"• {fee['label_en']}: Cat A {egp['A']['total']} | Cat B {egp['B']['total']} | Cat C {egp['C']['total']} | Cat D {egp['D']['total']}{uk}"
            )

    if is_ar:
        answer = (
            "فئات المصروفات الدراسية السنوية الرسمية بجامعة MSA للطلبة المصريين (2026/2027):\n"
            + "\n".join(lines)
            + "\n\nملاحظة: تُدفع رسوم الاعتماد البريطاني (£335 بيدفوردشير أو £440 جرينتش) في الفصل الدراسي الأول، وتخضع المصروفات لزيادة سنوية من 10% إلى 15%.\n\n"
            + "اختر أي كلية بالأسفل لعرض تفصيل الترم الأول والثاني ومصاريف الطلاب الأجانب بالدولار:"
        )
    else:
        answer = (
            "Official MSA University 2026/2027 Annual Tuition Fees Categories for Egyptians (EGP):\n"
            + "\n".join(lines)
            + "\n\nNote: British validation fees (£335 Bedfordshire or £440 Greenwich) are paid in the 1st semester each year, and full tuition is subject to a 10% to 15% annual increase.\n\n"
            + "Choose any faculty below to see the exact 1st & 2nd semester split plus International (USD) fees:"
        )

    faculty_options = [
        {
            "gate": "",
            "label": fee["label_ar"] if is_ar else fee["label_en"].replace("Faculty of ", ""),
            "target": "MSA_TUITION_FACULTY_SELECT",
            "query": f"مصاريف {fee['label_ar']}" if is_ar else f"{fee['label_en']} tuition fees",
        }
        for fee in MSA_TUITION_FEES.values()
    ]
    return {
        "answer": answer,
        "matches": [
            {
                "id": "msa_tuition_all_faculties",
                "kind": "tuition_table",
                "title": "MSA Official Tuition Fees Categories (2026/2027)",
                "excerpt": answer[:260],
                "source_file": "MSA_Tuition_Fees_and_Scholarships_Official.md",
                "course": "MSA Official Tuition Fees (2026/2027)",
                "confidence": 0.99,
            }
        ],
        "mode": "msa_tuition_guide",
        "needs_gate": True,
        "pending_target": "MSA_TUITION_FACULTY_SELECT",
        "gate_options": faculty_options,
    }


def resolve_msa_faculties(question: str, pending_target: str = "") -> Optional[Dict[str, Any]]:
    """3-step interactive MSA academic advisor: Faculties (bullets) -> Majors (bullets) -> Full 4/5 Year Plan (bullets)."""
    q_clean = clean_text(question)
    q_lang = detect_language(q_clean)
    is_ar = q_lang == "ar"
    q_tokens = _tokenize(q_clean)

    # Step 3: User is selecting a Major after seeing a faculty's majors
    if pending_target.startswith("MSA_MAJOR_SELECT:"):
        fac_key = pending_target.split(":", 1)[1]
        matched = _match_specific_major(q_clean, restrict_faculty=fac_key) or _match_specific_major(q_clean)
        if matched:
            return _format_major_study_plan_response(matched[0], matched[1], is_ar)

    # Step 2: User is selecting a Faculty after "Which faculty are you interested in?"
    if pending_target == "MSA_FACULTY_SELECT":
        matched_fac = _match_specific_faculty(q_clean)
        if matched_fac:
            return _format_faculty_majors_response(matched_fac, is_ar)
        matched_major = _match_specific_major(q_clean)
        if matched_major:
            return _format_major_study_plan_response(matched_major[0], matched_major[1], is_ar)

    # Direct request for a specific study plan (e.g., "CSE 5 year plan", "Mechatronics subjects", "خطة مواد الميكاترونكس")
    asks_plan = bool(
        re.search(r"\b(study plan|five year|5 year|four year|4 year|subjects|courses|curriculum|modules|syllabus)\b|(خطة|خطه|مواد|المواد|مقررات|المنهج)", q_clean, re.IGNORECASE)
    )
    if asks_plan:
        matched_major = _match_specific_major(q_clean)
        if matched_major:
            return _format_major_study_plan_response(matched_major[0], matched_major[1], is_ar)
        matched_fac = _match_specific_faculty(q_clean)
        if matched_fac:
            return _format_faculty_majors_response(matched_fac, is_ar)

    # Step 1: General "what faculties does MSA have?" (including typos like 'fucalties', 'faculities')
    asks_faculties_list = (
        _fuzzy_contains(q_tokens, ["faculties", "faculty", "colleges", "majors", "programs"], threshold=0.74)
        or any(tok in {"كليات", "كليه", "تخصصات", "اقسام"} for tok in q_tokens)
    )
    is_location_q = bool(
        re.search(r"\b(where|building|room|gate|floor)\b|(فين|أين|اين|مبنى|مبني|قاعة|قاعه|غرفة|غرفه|بوابة|بوابه)", q_clean, re.IGNORECASE)
    )
    is_rules_q = bool(
        re.search(
            r"\b(attendance|gpa|probation|document|documents|igcse|american|elpt|exam|plagiarism|bus|greenwich|bedfordshire|tuition|fee|fees|cost|price|scholarship|scholarships|discount|minimum|score|scores|categories|category|refund|payment|transfer)\b|(غياب|حضور|معدل|اوراق|قبول|امتحان|اتوبيس|جرينتش|بيدفوردشير|مصاريف|مصروفات|رسوم|سعر|بكام|كام|فلوس|منح|منحة|خصم|تنسيق|الحد الادنى|الحد الأدنى|تحويل|سداد|دفع)",
            q_clean,
            re.IGNORECASE,
        )
    )

    if asks_faculties_list and not is_location_q and not is_rules_q:
        specific_fac = _match_specific_faculty(q_clean)
        if specific_fac:
            return _format_faculty_majors_response(specific_fac, is_ar)

        if is_ar:
            bullets = "\n".join(f"• {info['label_ar']} ({info['duration_ar']})" for info in MSA_FACULTIES_DATA.values())
            answer = f"تضم جامعة MSA إحدى عشرة كلية رئيسية:\n{bullets}\n\nما هي الكلية التي تهتم بها؟"
        else:
            bullets = "\n".join(f"• {info['label_en']} ({info['duration_en']})" for info in MSA_FACULTIES_DATA.values())
            answer = f"MSA University offers 11 main faculties:\n{bullets}\n\nWhich faculty are you interested in?"

        faculty_options = [
            {
                "gate": "",
                "label": info["label_ar"] if is_ar else info["label_en"],
                "target": "MSA_FACULTY_SELECT",
                "query": info["label_ar"] if is_ar else info["label_en"],
            }
            for info in MSA_FACULTIES_DATA.values()
        ]
        return {
            "answer": answer,
            "matches": [
                {
                    "id": "msa_faculties_overview",
                    "kind": "faculty_guide",
                    "title": "MSA University Faculties Overview",
                    "excerpt": answer[:260],
                    "source_file": "MSA_Study_Plans_All_Faculties.md",
                    "course": "MSA University Study Plans",
                    "confidence": 0.99,
                }
            ],
            "mode": "msa_faculty_guide",
            "needs_gate": True,
            "pending_target": "MSA_FACULTY_SELECT",
            "gate_options": faculty_options,
        }

    return None


def build_contextual_followups(question: str, answer: str, mode: str, is_ar: bool) -> List[str]:
    """Generate 3 context-aware follow-up questions based on the student's question."""
    q_low = clean_text(question).lower()
    combined = f"{q_low} {clean_text(answer).lower()}"

    if any(k in combined for k in ("8085", "8255", "8259", "arm", "i2c", "microprocessor", "assembly", "interrupt", "ppi", "pic", "register", "معالج", "اسمبلي")):
        pool = (
            [
                "اشرح كيف يعمل متحكم المقاطعات PIC 8259 في معالج 8085؟",
                "ما هي أوضاع التشغيل في شريحة PPI 8255؟",
                "كيف يعمل بروتوكول الاتصال التسلسلي I2C؟",
                "ما الفرق بين برمجة معالج 8085 ومعالج ARM Assembly؟",
            ]
            if is_ar
            else [
                "How does the PPI 8255 work with the 8085 microprocessor?",
                "How do PIC 8259 interrupts work?",
                "Explain I2C serial communication from the Microprocessors lectures.",
                "What are the main instructions in ARM Assembly from the Micro labs?",
            ]
        )
    elif any(k in combined for k in ("neural network", "gradient descent", "confusion matrix", "svm", "clustering", "k-means", "kmeans", "q-learning", "qlearning", "reinforcement", "intelligent agent", "search technique")):
        pool = (
            [
                "ما هو الـ Confusion Matrix وكيف نحسب الدقة والـ Recall؟",
                "اشرح خوارزمية Gradient Descent في الشبكات العصبية؟",
                "كيف تعمل خوارزمية K-Means Clustering؟",
                "ما هو التعلم بالتعزيز (Reinforcement Learning) و Q-Learning؟",
            ]
            if is_ar
            else [
                "Explain Gradient Descent optimization from the AI lectures.",
                "How does SVM classification work in the AI course?",
                "Explain K-Means clustering and how it works.",
                "How does Q-Learning work in Reinforcement Learning?",
            ]
        )
    elif any(k in combined for k in ("food court", "krispy kreme", "qadra", "restaurant", "cafeteria", "فود كورت", "مطاعم", "قدرة", "كريسبي")):
        pool = (
            [
                "كيف تُدفع المصروفات الدراسية ورسوم الاعتماد البريطاني في MSA؟",
                "ما هي المنح الدراسية والخصومات المتاحة في جامعة MSA؟",
                "فين غرفة K108 مكتب اشتراكات الأتوبيسات؟",
                "فين المكتبة الرئيسية وجيم الجامعة؟",
            ]
            if is_ar
            else [
                "How do tuition fees, British validation fees, and annual increases work at MSA University?",
                "What scholarships and tuition discounts does MSA University offer?",
                "Where is room K108?",
                "Where is the Main Library and MSA Gym?",
            ]
        )
    elif mode == "msa_room_locator" or any(k in q_low for k in ("room", "building", "gate", "architecture", "opera", "library", "bus", "قاعة", "غرفة", "مبنى", "بوابة", "عمارة", "مكتبة")):
        pool = (
            [
                "إيه المطاعم الموجودة في الفود كورت؟",
                "كيف تُدفع المصروفات الدراسية ورسوم الاعتماد البريطاني في MSA؟",
                "ما هي المنح الدراسية والخصومات المتاحة في جامعة MSA؟",
                "ما هي كليات جامعة MSA؟",
            ]
            if is_ar
            else [
                "What restaurants are in the food court?",
                "How do tuition fees, British validation fees, and annual increases work at MSA University?",
                "What scholarships and tuition discounts does MSA University offer?",
                "What are the faculties MSA University has?",
            ]
        )
    elif any(k in q_low for k in ("attendance", "gpa", "probation", "exam", "plagiarism", "credit", "rules", "غياب", "حضور", "معدل", "امتحان", "انذار", "لوائح")):
        pool = (
            [
                "كيف تُدفع المصروفات الدراسية ورسوم الاعتماد البريطاني في MSA؟",
                "ما هي المنح الدراسية والخصومات المتاحة في جامعة MSA؟",
                "ما هي قواعد التأخير في الميدتيرم والامتحان النهائي؟",
                "كيف تعمل الشهادة البريطانية المزدوجة في MSA؟",
            ]
            if is_ar
            else [
                "How do tuition fees, British validation fees, and annual increases work at MSA University?",
                "What scholarships and tuition discounts does MSA University offer?",
                "What are the late arrival rules for midterm and final exams at MSA University?",
                "How does the MSA British Dual Degree work and is equivalency needed in the UK?",
            ]
        )
    elif any(k in q_low for k in ("admission", "document", "igcse", "american", "thanaweya", "elpt", "تقديم", "قبول", "اوراق", "ثانوية")):
        pool = (
            [
                "كيف تُدفع المصروفات الدراسية ورسوم الاعتماد البريطاني في MSA؟",
                "ما هي المنح الدراسية والخصومات المتاحة في جامعة MSA؟",
                "ما هي كليات جامعة MSA؟",
                "هل توجد أتوبيسات للجامعة وفين مكتب الاشتراك؟",
            ]
            if is_ar
            else [
                "How do tuition fees, British validation fees, and annual increases work at MSA University?",
                "What scholarships and tuition discounts does MSA University offer?",
                "What are the faculties MSA University has?",
                "Does MSA University provide campus buses and where do I subscribe?",
            ]
        )
    elif any(k in q_low for k in ("tuition", "scholarship", "discount", "dual degree", "syndicate", "summer abroad", "transfer", "مصروفات", "منح", "خصم", "شهادة", "نقابات", "صيفي", "تحويل")):
        pool = (
            [
                "ما هي المنح الدراسية والخصومات المتاحة في جامعة MSA؟",
                "كيف تُدفع المصروفات الدراسية ورسوم الاعتماد البريطاني في MSA؟",
                "هل خريجو جامعة MSA مؤهلون للانضمام للنقابات المهنية المصرية؟",
                "إيه الأوراق المطلوبة للتقديم في جامعة MSA؟",
            ]
            if is_ar
            else [
                "What scholarships and tuition discounts does MSA University offer?",
                "How do tuition fees, British validation fees, and annual increases work at MSA University?",
                "Are MSA University graduates eligible to join Egyptian Professional Syndicates?",
                "What general documents are required for admission to MSA University?",
            ]
        )
    else:
        pool = (
            [
                "كيف تُدفع المصروفات الدراسية ورسوم الاعتماد البريطاني في MSA؟",
                "ما هي المنح الدراسية والخصومات المتاحة في جامعة MSA؟",
                "ما هي كليات جامعة MSA؟",
                "إيه الأوراق المطلوبة للتقديم في جامعة MSA؟",
            ]
            if is_ar
            else [
                "How do tuition fees, British validation fees, and annual increases work at MSA University?",
                "What scholarships and tuition discounts does MSA University offer?",
                "What are the faculties MSA University has?",
                "What general documents are required for admission to MSA University?",
            ]
        )

    # Filter out any suggestion too similar to the user's current question and take exactly 3
    q_toks = set(_tokenize(question))
    picked: List[str] = []
    for candidate in pool:
        cand_toks = set(_tokenize(candidate))
        overlap = len(q_toks & cand_toks) / max(1, min(len(q_toks), len(cand_toks))) if q_toks and cand_toks else 0.0
        if overlap >= 0.72:
            continue
        picked.append(candidate)
        if len(picked) == 3:
            break

    # Ensure always 3 suggestions if filtering removed too many
    for candidate in pool:
        if len(picked) == 3:
            break
        if candidate not in picked:
            picked.append(candidate)
    return picked[:3]


def _build_not_in_records_response(question: str, is_ar: bool, rag_trace: Dict[str, Any], latency_ms: float) -> Dict[str, Any]:
    """Standard anti-hallucination refusal + interactive offer to guide the student to Student Affairs."""
    if is_ar:
        answer = (
            "لا تتوفر هذه المعلومة في سجلات الكلية لدي.\n"
            "يرجى مراجعة شؤون الطلاب.\n\n"
            "هل تريد أن أرشدك إلى موقع شؤون الطلاب؟"
        )
        options = [
            {"gate": "", "label": "نعم، أرشدني لموقع شؤون الطلاب", "target": "GUIDE_STUDENT_AFFAIRS", "query": "نعم"},
            {"gate": "", "label": "لا، شكراً", "target": "GUIDE_STUDENT_AFFAIRS", "query": "لا"},
        ]
    else:
        answer = (
            "I don't have that information in my faculty records.\n"
            "Please check with Student Affairs.\n\n"
            "Would you like me to guide you to the Student Affairs location?"
        )
        options = [
            {"gate": "", "label": "Yes, guide me to Student Affairs", "target": "GUIDE_STUDENT_AFFAIRS", "query": "Yes"},
            {"gate": "", "label": "No, thanks", "target": "GUIDE_STUDENT_AFFAIRS", "query": "No"},
        ]
    return {
        "answer": answer,
        "matches": [],
        "latency_ms": latency_ms,
        "mode": "rag_not_in_records",
        "needs_gate": True,
        "pending_target": "GUIDE_STUDENT_AFFAIRS",
        "gate_options": options,
        "rag_trace": rag_trace,
        "suggested_questions": [],
    }


def query_local_rag(
    question: str,
    chunks: List[Dict[str, Any]],
    qa_pairs: List[Dict[str, Any]],
    top_k: int = 3,
    use_ollama: bool = False,
    ollama_model: str = "phi3:mini",
    entry_gate: str = "",
    pending_target: str = "",
    rag_engine: Any = None,
) -> Dict[str, Any]:
    """Run real Vector RAG (ChromaDB embeddings + retrieval + augmented generation) with interactive MSA dialogs."""
    start = time.perf_counter()

    # Stage 1: Real Vector Retrieval from ChromaDB (or local cosine vector index)
    retrieved_vectors: List[Dict[str, Any]] = []
    rag_status: Dict[str, Any] = {
        "vector_db": "Local Cosine Vector Store",
        "embedding_model": "Bilingual Hybrid Vector (384d)",
        "indexed_vectors": len(chunks) + len(qa_pairs),
    }
    if rag_engine is not None and (chunks or qa_pairs):
        retrieved_vectors, rag_status = rag_engine.retrieve(question, chunks, qa_pairs, top_k=max(top_k, 5))

    def _build_rag_trace(generator_label: str, fallback_matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        trace_chunks = [
            {
                "id": item["id"],
                "source_file": item["source_file"],
                "heading": item["title"],
                "similarity": round(item.get("vector_similarity", item.get("confidence", 0.0)), 2),
                "excerpt": item["display_text"][:180],
            }
            for item in retrieved_vectors[:top_k]
        ] or [
            {
                "id": m["id"],
                "source_file": m["source_file"],
                "heading": m["title"],
                "similarity": m.get("confidence", 0.95),
                "excerpt": m.get("excerpt", "")[:180],
            }
            for m in fallback_matches[:top_k]
        ]
        return {
            "vector_db": rag_status.get("vector_db", "ChromaDB (Persistent)"),
            "embedding_model": rag_status.get("embedding_model", "Bilingual Hybrid Vector (384d)"),
            "indexed_vectors": rag_status.get("indexed_vectors", len(chunks) + len(qa_pairs)),
            "generator": generator_label,
            "retrieved_chunks": trace_chunks,
        }

    is_ar_q = detect_language(question) == "ar"

    # Handle Yes/No follow-up to "Would you like me to guide you to the Student Affairs location?"
    if pending_target == "GUIDE_STUDENT_AFFAIRS":
        q_low = clean_text(question).lower()
        is_yes = bool(re.search(r"\b(yes|yeah|yep|sure|ok|okay|please|guide|show|student affairs)\b|(نعم|ايوه|أيوة|اه|آه|أرشدني|موافق|اكيد|أكيد)", q_low, re.IGNORECASE))
        is_no = bool(re.search(r"\b(no|nope|nah|cancel|thanks)\b|(لا|لأ|شكراً|شكرا)", q_low, re.IGNORECASE)) and not is_yes
        if is_yes:
            bldg = MSA_BUILDINGS["SSB"]
            header = (
                f"تقع مكاتب شؤون الطلاب في {bldg['ar_name']}.\n\nمن أي بوابة دخلت الجامعة؟ اختر بوابة 1 أو بوابة 2 أو بوابة 3 أو بوابة 4 لأعطيك الاتجاهات."
                if is_ar_q
                else f"Student Affairs is located in {bldg['en_name']}.\n\nWhich gate did you enter from? Choose Gate 1, Gate 2, Gate 3, or Gate 4 for directions."
            )
            return {
                "answer": header,
                "matches": [
                    {
                        "id": "msa_bldg_ssb",
                        "kind": "room_locator",
                        "title": bldg["ar_name"] if is_ar_q else bldg["en_name"],
                        "excerpt": header[:220],
                        "source_file": "MSA_Rooms_and_Locations_Guide.md",
                        "course": "MSA Campus Map & Gate Directions",
                        "confidence": 0.99,
                    }
                ],
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
                "mode": "msa_room_locator",
                "needs_gate": True,
                "pending_target": "SSB",
                "gate_options": _build_gate_options("SSB", is_ar_q),
                "rag_trace": _build_rag_trace("RAG + Student Affairs Wayfinding", []),
                "suggested_questions": [],
            }
        if is_no:
            polite = (
                "لا مشكلة. يمكنك سؤالي عن أي شيء آخر يخص جامعة MSA."
                if is_ar_q
                else "No problem. Feel free to ask me anything else about MSA University."
            )
            return {
                "answer": polite,
                "matches": [],
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
                "mode": "assistant_ack",
                "needs_gate": False,
                "pending_target": "",
                "gate_options": [],
                "rag_trace": _build_rag_trace("Campus Assistant", []),
                "suggested_questions": build_contextual_followups("general", polite, "assistant_ack", is_ar_q),
            }

    # Check faculty-specific Q1–Q5 FAQ handler (when user picked a faculty study plan and clicked/typed Q1–Q5)
    faq_hit = resolve_msa_faculty_faq(question, pending_target=pending_target)
    if faq_hit:
        faq_hit["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
        faq_hit["rag_trace"] = _build_rag_trace("RAG + Faculty-Specific MSA FAQ", faq_hit["matches"])
        faq_hit["suggested_questions"] = build_contextual_followups(question, faq_hit["answer"], faq_hit["mode"], is_ar_q)
        return faq_hit

    # Check MSA official 2026/2027 tuition fees categories advisor (all 11 faculties x A/B/C/D)
    tuition_hit = resolve_msa_tuition_fees(question, pending_target=pending_target)
    if tuition_hit:
        tuition_hit["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
        tuition_hit["rag_trace"] = _build_rag_trace("RAG + Official MSA Tuition Table (2026/2027)", tuition_hit["matches"])
        if not tuition_hit.get("needs_gate") and not tuition_hit.get("gate_options"):
            tuition_hit["suggested_questions"] = build_contextual_followups(question, tuition_hit["answer"], tuition_hit["mode"], is_ar_q)
        else:
            tuition_hit["suggested_questions"] = []
        return tuition_hit

    # Check interactive MSA faculties guide (augmented with live vector RAG trace)
    faculty_hit = resolve_msa_faculties(question, pending_target=pending_target)
    if faculty_hit:
        faculty_hit["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
        faculty_hit["rag_trace"] = _build_rag_trace("RAG + Interactive Faculty Dialog", faculty_hit["matches"])
        if not faculty_hit.get("needs_gate") and not faculty_hit.get("gate_options"):
            faculty_hit["suggested_questions"] = build_contextual_followups(question, faculty_hit["answer"], faculty_hit["mode"], is_ar_q)
        else:
            faculty_hit["suggested_questions"] = []
        return faculty_hit

    # Check MSA room / building / facility locator (augmented with live vector RAG trace)
    room_hit = resolve_msa_room_or_location(question, chunks, entry_gate=entry_gate, pending_target=pending_target)
    if room_hit:
        room_hit["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
        room_hit["rag_trace"] = _build_rag_trace("RAG + MSA Campus Gate Locator", room_hit["matches"])
        room_hit["suggested_questions"] = (
            [] if room_hit.get("needs_gate") else build_contextual_followups(question, room_hit["answer"], room_hit["mode"], is_ar_q)
        )
        return room_hit

    if not chunks and not qa_pairs:
        return {
            "answer": "No course or faculty documents are indexed yet. Upload files or load the MSA University dataset first.",
            "matches": [],
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            "mode": "empty",
            "rag_trace": _build_rag_trace("Empty index", []),
            "suggested_questions": [],
        }

    # Stage 2: Vector RAG Generation (ChromaDB retrieved chunks -> Ollama LLM or Grounded Synthesis)
    if rag_engine is not None and retrieved_vectors:
        answer, gen_mode = rag_engine.generate_augmented_answer(
            question=question,
            retrieved=retrieved_vectors,
            use_ollama=use_ollama,
            ollama_model=ollama_model,
        )
        if gen_mode == "not_found" or answer == "NOT_IN_RECORDS":
            return _build_not_in_records_response(
                question,
                is_ar_q,
                _build_rag_trace("Anti-Hallucination Guardrail (Not in Records)", []),
                round((time.perf_counter() - start) * 1000, 1),
            )
        formatted_matches = [
            {
                "id": item["id"],
                "kind": item["kind"],
                "title": item["title"],
                "excerpt": item["display_text"][:260],
                "source_file": item["source_file"],
                "course": item["course"],
                "confidence": item["confidence"],
            }
            for item in retrieved_vectors[:top_k]
        ]
        gen_label = f"Ollama {ollama_model} (RAG)" if "ollama" in gen_mode else "ChromaDB Grounded RAG Synthesizer"
        return {
            "answer": answer,
            "matches": formatted_matches,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            "mode": gen_mode,
            "rag_trace": _build_rag_trace(gen_label, formatted_matches),
            "suggested_questions": build_contextual_followups(question, answer, gen_mode, is_ar_q),
        }

    # Fallback lexical BM25 retrieval if rag_engine was not provided
    query_tokens = _tokenize(question)
    if not query_tokens:
        return {
            "answer": "Please ask a question in Arabic or English.",
            "matches": [],
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            "mode": "empty",
            "rag_trace": _build_rag_trace("No query tokens", []),
        }

    corpus: List[Dict[str, Any]] = []
    for qa in qa_pairs:
        corpus.append(
            {
                "kind": "qa",
                "id": qa["id"],
                "title": qa["question"],
                "text": f"{qa['question']} {qa['answer']}",
                "display_text": qa["answer"],
                "source_file": qa["source_file"],
                "course": qa["course"],
                "language": qa["language"],
            }
        )
    for chk in chunks:
        corpus.append(
            {
                "kind": "chunk",
                "id": chk["id"],
                "title": chk["heading"],
                "text": f"{chk['heading']} {chk['text']}",
                "display_text": chk["text"],
                "source_file": chk["source_file"],
                "course": chk["course"],
                "language": chk["language"],
            }
        )

    doc_tokens = [_tokenize(doc["text"]) for doc in corpus]
    df: Counter[str] = Counter()
    for toks in doc_tokens:
        for t in set(toks):
            df[t] += 1

    n_docs = len(corpus)
    q_counts = Counter(query_tokens)
    scored: List[Tuple[float, Dict[str, Any]]] = []

    for doc, toks in zip(corpus, doc_tokens):
        if not toks:
            continue
        t_counts = Counter(toks)
        score = 0.0
        for tok, qf in q_counts.items():
            tf = t_counts.get(tok, 0)
            if tf == 0:
                continue
            idf = math.log(1 + (n_docs - df[tok] + 0.5) / (df[tok] + 0.5))
            score += idf * ((tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * len(toks) / 45.0))) * qf
        if doc["kind"] == "qa" and score > 0:
            score *= 1.15
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    top_matches = scored[:top_k]

    if not top_matches:
        return _build_not_in_records_response(
            question,
            is_ar_q,
            _build_rag_trace("Anti-Hallucination Guardrail (Not in Records)", []),
            round((time.perf_counter() - start) * 1000, 1),
        )

    context_blocks = "\n\n".join(
        f"[{m['source_file']} · {m['title']}]\n{m['display_text']}" for _, m in top_matches
    )
    mode = "local_bm25"
    primary_text = top_matches[0][1]["display_text"]
    answer = primary_text
    if len(top_matches) > 1 and top_matches[0][0] > 0:
        second_score, second_doc = top_matches[1]
        second_text = second_doc["display_text"]
        if (
            second_score >= top_matches[0][0] * 0.72
            and second_text not in primary_text
            and primary_text not in second_text
            and len(primary_text) + len(second_text) < 1200
        ):
            answer = f"{primary_text}\n\n{second_text}"

    if use_ollama and check_ollama_status().get("available"):
        try:
            prompt = (
                "You are an offline university campus assistant. Answer the student's question "
                "using only the provided context. Reply in the same language as the student "
                "(Egyptian Arabic or English). Be clear and concise.\n\n"
                f"Context:\n{context_blocks}\n\nStudent question: {question}\nAnswer:"
            )
            resp = requests.post(
                "http://127.0.0.1:11434/api/generate",
                json={"model": ollama_model, "prompt": prompt, "stream": False},
                timeout=20,
            )
            if resp.status_code == 200 and resp.json().get("response"):
                answer = clean_text(resp.json()["response"])
                mode = f"ollama:{ollama_model}"
        except Exception:
            pass

    max_score = top_matches[0][0] or 1.0
    formatted_matches = [
        {
            "id": m["id"],
            "kind": m["kind"],
            "title": m["title"],
            "excerpt": m["display_text"][:260],
            "source_file": m["source_file"],
            "course": m["course"],
            "confidence": round(min(0.99, score / (max_score + 0.35)), 2),
        }
        for score, m in top_matches
    ]

    return {
        "answer": answer,
        "matches": formatted_matches,
        "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        "mode": mode,
    }


def check_ollama_status(ollama_url: str = "http://127.0.0.1:11434") -> Dict[str, Any]:
    """Probe local Ollama daemon without blocking."""
    try:
        resp = requests.get(f"{ollama_url}/api/tags", timeout=1.2)
        if resp.status_code == 200:
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            return {"available": True, "models": models}
    except Exception:
        pass
    return {"available": False, "models": []}


def check_chroma_status(chroma_url: str = "http://127.0.0.1:8000") -> Dict[str, Any]:
    """Check whether Python chromadb or a local Chroma HTTP server is available."""
    try:
        import chromadb  # type: ignore  # noqa: F401

        return {"available": True, "mode": "python_embedded"}
    except ImportError:
        pass
    try:
        resp = requests.get(f"{chroma_url}/api/v1/heartbeat", timeout=1.0)
        if resp.status_code == 200:
            return {"available": True, "mode": "http_server"}
    except Exception:
        pass
    return {"available": False, "mode": "export_ready"}


def build_chromadb_payload(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a ChromaDB-ready collection payload for nomic-embed-text ingestion."""
    return {
        "collection_name": "campus_assistant_knowledge",
        "embedding_model": "nomic-embed-text",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_chunks": len(chunks),
        "ids": [c["id"] for c in chunks],
        "documents": [c["text"] for c in chunks],
        "metadatas": [
            {
                "heading": c["heading"],
                "source_file": c["source_file"],
                "location": c["location"],
                "course": c["course"],
                "department": c["department"],
                "language": c["language"],
                "token_estimate": c["token_estimate"],
            }
            for c in chunks
        ],
    }


def export_jsonl_bytes(qa_pairs: List[Dict[str, Any]]) -> bytes:
    """Export Q&A pairs as UTF-8 JSONL suitable for RAG evaluation and LLM instruction tuning."""
    lines = []
    for item in qa_pairs:
        record = {
            "id": item["id"],
            "instruction": item["question"],
            "response": item["answer"],
            "metadata": {
                "topic": item["topic"],
                "source_file": item["source_file"],
                "course": item["course"],
                "department": item["department"],
                "language": item["language"],
            },
        }
        lines.append(json.dumps(record, ensure_ascii=False))
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def export_csv_bytes(chunks: List[Dict[str, Any]], qa_pairs: List[Dict[str, Any]]) -> bytes:
    """Export combined dataset as UTF-8 BOM CSV for Arabic-safe spreadsheet compatibility."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["record_type", "id", "course", "department", "language", "source_file", "title_or_question", "content_or_answer", "token_estimate"])
    for qa in qa_pairs:
        writer.writerow(["qa_pair", qa["id"], qa["course"], qa["department"], qa["language"], qa["source_file"], qa["question"], qa["answer"], estimate_tokens(qa["answer"])])
    for chk in chunks:
        writer.writerow(["rag_chunk", chk["id"], chk["course"], chk["department"], chk["language"], chk["source_file"], chk["heading"], chk["text"], chk["token_estimate"]])
    return ("\ufeff" + buf.getvalue()).encode("utf-8")


def export_bundle_zip_bytes(chunks: List[Dict[str, Any]], qa_pairs: List[Dict[str, Any]], ingest_script_path: Path) -> bytes:
    """Bundle JSON, JSONL, CSV, ChromaDB ingestion script, and Raspberry Pi edge deploy scripts into a ZIP."""
    mem = io.BytesIO()
    scripts_dir = ingest_script_path.parent
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "rag_chunks_chromadb.json",
            json.dumps(build_chromadb_payload(chunks), ensure_ascii=False, indent=2),
        )
        zf.writestr("qa_dataset.jsonl", export_jsonl_bytes(qa_pairs))
        zf.writestr("campus_dataset.csv", export_csv_bytes(chunks, qa_pairs))
        for script_name in ("ingest_chromadb.py", "deploy_raspberry_pi.sh", "kiosk_start.sh"):
            fpath = scripts_dir / script_name
            if fpath.exists():
                zf.writestr(script_name, fpath.read_text(encoding="utf-8"))
    return mem.getvalue()


def ingest_into_local_chroma(chunks: List[Dict[str, Any]], persist_dir: Path) -> Dict[str, Any]:
    """Ingest chunks into local ChromaDB using Ollama nomic-embed-text when available."""
    if not chunks:
        return {"ok": False, "message": "Upload course files before syncing to ChromaDB."}

    chroma_info = check_chroma_status()
    if not chroma_info["available"]:
        return {
            "ok": False,
            "message": "Local ChromaDB is not installed in this Python environment. Export the Pi bundle ZIP and run ingest_chromadb.py on your Raspberry Pi.",
        }

    ollama_info = check_ollama_status()
    embeddings: Optional[List[List[float]]] = None
    if ollama_info["available"] and any("nomic-embed-text" in m for m in ollama_info["models"]):
        embeddings = []
        for c in chunks:
            r = requests.post(
                "http://127.0.0.1:11434/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": c["text"]},
                timeout=15,
            )
            if r.status_code != 200 or "embedding" not in r.json():
                embeddings = None
                break
            embeddings.append(r.json()["embedding"])

    try:
        import chromadb  # type: ignore

        client = chromadb.PersistentClient(path=str(persist_dir))
        collection = client.get_or_create_collection(name="campus_assistant_knowledge")
        payload = build_chromadb_payload(chunks)
        kwargs: Dict[str, Any] = {
            "ids": payload["ids"],
            "documents": payload["documents"],
            "metadatas": payload["metadatas"],
        }
        if embeddings:
            kwargs["embeddings"] = embeddings
        collection.upsert(**kwargs)
        embed_note = "nomic-embed-text via Ollama" if embeddings else "ChromaDB default local embeddings"
        return {
            "ok": True,
            "message": f"Synced {len(chunks)} chunks into local ChromaDB using {embed_note}.",
            "count": len(chunks),
        }
    except Exception as exc:
        return {"ok": False, "message": f"ChromaDB sync failed. {exc}"}


def get_sample_documents() -> List[Dict[str, Any]]:
    """Official bilingual MSA University (6th of October) knowledge base files."""
    knowledge_dir = Path(__file__).resolve().parent.parent / "data" / "msa_knowledge"
    specs = [
        ("MSA_Overview_and_Faculties_EN.md", "MSA University Overview & Faculties", "Student Affairs"),
        ("MSA_Academic_Rules_and_Admission_EN.md", "MSA Academic Rules & Admission Bylaws", "Academic Regulations"),
        ("MSA_Tuition_Fees_and_Scholarships_Official.md", "MSA Official Tuition Fees (2026/2027), Scholarships & Bus Fees", "Student Affairs"),
        ("MSA_Admission_Guide_Official.md", "MSA Official Admission Guide, Min Scores, ELPT & Documents", "Student Affairs"),
        ("MSA_Campus_Buildings_and_Architecture_EN.md", "MSA Campus Buildings & Architecture", "Engineering & Embedded"),
        ("MSA_Rooms_and_Locations_Guide.md", "MSA Rooms & Building Locator Guide", "Student Affairs"),
        ("MSA_Food_Court_and_Dining.md", "MSA Food Court, Cafeterias & Dining", "Student Affairs"),
        ("MSA_Top_5_Internet_FAQs.md", "MSA Top 5 Internet FAQs (Bilingual)", "Student Affairs"),
        ("MSA_University_Guide_AR.md", "دليل جامعة MSA والمباني واللوائح", "Student Affairs"),
    ]
    docs: List[Dict[str, Any]] = []
    for filename, course, department in specs:
        fpath = knowledge_dir / filename
        if fpath.exists():
            docs.append(
                {
                    "filename": filename,
                    "course": course,
                    "department": department,
                    "content": fpath.read_bytes(),
                }
            )
    docs.append(
        {
            "filename": "MSA_Study_Plans_All_Faculties.md",
            "course": "MSA 4 & 5 Year Study Plans (All 11 Faculties)",
            "department": "Academic Regulations",
            "content": build_study_plans_markdown().encode("utf-8"),
        }
    )
    return docs


def build_from_sample_documents(chunk_size: int = 512, chunk_overlap: int = 96) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Process the built-in bilingual sample files into chunks, Q&A pairs, and source summaries."""
    all_chunks: List[Dict[str, Any]] = []
    all_qa: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []

    for doc in get_sample_documents():
        sections = parse_uploaded_file(doc["filename"], doc["content"])
        chunks = chunk_sections(
            sections,
            source_file=doc["filename"],
            course_name=doc["course"],
            department=doc["department"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        qa_pairs, _ = extract_qa_pairs(chunks, use_ollama=False)
        all_chunks.extend(chunks)
        all_qa.extend(qa_pairs)
        sources.append(
            {
                "filename": doc["filename"],
                "course": doc["course"],
                "department": doc["department"],
                "chunks_count": len(chunks),
                "qa_count": len(qa_pairs),
                "language": detect_language(" ".join(c["text"] for c in chunks)),
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    return all_chunks, all_qa, sources
