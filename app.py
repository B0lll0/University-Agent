"""Flask server for the Offline Edge AI Campus Assistant Dataset Portal."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

from backend.dataset_builder import (
    build_chromadb_payload,
    build_from_sample_documents,
    check_chroma_status,
    check_ollama_status,
    chunk_sections,
    estimate_tokens,
    export_bundle_zip_bytes,
    export_csv_bytes,
    export_jsonl_bytes,
    extract_qa_pairs,
    ingest_into_local_chroma,
    query_local_rag,
)
from backend.parsers import ALLOWED_EXTENSIONS, clean_text, detect_language, parse_uploaded_file
from backend.rag_engine import EdgeRAGEngine
from backend.tts_engine import get_tts_capabilities, synthesize_wav

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STORE_PATH = DATA_DIR / "dataset_store.json"
CHROMA_DIR = DATA_DIR / "chroma_db"
INGEST_SCRIPT_PATH = BASE_DIR / "scripts" / "ingest_chromadb.py"

MAX_FILE_BYTES = 20 * 1024 * 1024  # 20 MB per file

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
app.config["JSON_AS_ASCII"] = False
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

ASSET_VERSION = str(int(datetime.now(timezone.utc).timestamp()))
RAG_ENGINE = EdgeRAGEngine(CHROMA_DIR)


@app.after_request
def _no_cache_headers(response: Response) -> Response:
    if request.path.startswith("/static/") or request.path in {"/", "/kiosk", "/edge"}:
        response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


def _empty_store() -> Dict[str, Any]:
    return {
        "sources": [],
        "chunks": [],
        "qa_pairs": [],
        "updated_at": None,
    }


def load_store(auto_seed: bool = True) -> Dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not STORE_PATH.exists():
        if auto_seed:
            chunks, qa_pairs, sources = build_from_sample_documents(chunk_size=768, chunk_overlap=96)
            seeded = {"sources": sources, "chunks": chunks, "qa_pairs": qa_pairs}
            save_store(seeded)
            return seeded
        return _empty_store()
    try:
        return json.loads(STORE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return _empty_store()


def save_store(store: Dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    store["updated_at"] = datetime.now(timezone.utc).isoformat()
    STORE_PATH.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    RAG_ENGINE.rebuild_index(store.get("chunks", []), store.get("qa_pairs", []))


def _safe_filename(original_name: str) -> str:
    """Preserve Arabic filenames while stripping path traversal characters."""
    name = Path(original_name).name.strip()
    name = name.replace("/", "_").replace("\\", "_").replace("..", "")
    if not name or name.startswith("."):
        return secure_filename(original_name) or "uploaded_document.txt"
    return name


def _compute_stats(store: Dict[str, Any]) -> Dict[str, Any]:
    chunks = store.get("chunks", [])
    qa_pairs = store.get("qa_pairs", [])
    sources = store.get("sources", [])
    ar_chunks = sum(1 for c in chunks if c.get("language") in {"ar", "bilingual"})
    en_chunks = sum(1 for c in chunks if c.get("language") in {"en", "bilingual"})
    total_tokens = sum(int(c.get("token_estimate", 0)) for c in chunks)
    return {
        "sources_count": len(sources),
        "chunks_count": len(chunks),
        "qa_count": len(qa_pairs),
        "total_tokens": total_tokens,
        "arabic_chunks": ar_chunks,
        "english_chunks": en_chunks,
        "updated_at": store.get("updated_at"),
    }


@app.route("/")
def index() -> str:
    return render_template("index.html", asset_v=ASSET_VERSION)


@app.route("/edge")
@app.route("/kiosk")
def edge_kiosk_page() -> str:
    return render_template("edge.html", asset_v=ASSET_VERSION)


@app.route("/privacy")
def privacy_page() -> str:
    return render_template("legal.html", page_kind="privacy")


@app.route("/terms")
def terms_page() -> str:
    return render_template("legal.html", page_kind="terms")


@app.errorhandler(404)
def not_found(_err: Any) -> tuple[str, int]:
    return render_template("404.html"), 404


@app.route("/api/health", methods=["GET"])
def api_health() -> Response:
    store = load_store()
    if RAG_ENGINE.status["indexed_vectors"] != len(store.get("chunks", [])) + len(store.get("qa_pairs", [])):
        RAG_ENGINE.rebuild_index(store.get("chunks", []), store.get("qa_pairs", []))
    return jsonify(
        {
            "status": "ok",
            "offline_ready": True,
            "ollama": check_ollama_status(),
            "chromadb": check_chroma_status(),
            "rag": RAG_ENGINE.status,
            "tts": get_tts_capabilities(),
            "stats": _compute_stats(store),
        }
    )


@app.route("/api/tts", methods=["POST"])
def api_tts() -> Response | tuple[Response, int]:
    body = request.get_json(silent=True) or {}
    text = clean_text(str(body.get("text", "")))
    if not text:
        return jsonify({"error": "No text provided for speech synthesis."}), 400
    lang = str(body.get("lang", "")).strip() or detect_language(text)
    try:
        rate = float(body.get("rate", 1.0))
    except (TypeError, ValueError):
        rate = 1.0
    wav_bytes = synthesize_wav(text, lang=lang, rate=rate)
    if not wav_bytes:
        return jsonify({"error": "Offline server TTS is unavailable for this voice. Browser speech synthesis will be used instead."}), 503
    return Response(wav_bytes, mimetype="audio/wav")


@app.route("/api/dataset", methods=["GET"])
def api_get_dataset() -> Response:
    store = load_store()
    return jsonify(
        {
            "sources": store.get("sources", []),
            "chunks": store.get("chunks", []),
            "qa_pairs": store.get("qa_pairs", []),
            "stats": _compute_stats(store),
        }
    )


@app.route("/api/upload", methods=["POST"])
def api_upload() -> tuple[Response, int]:
    files = request.files.getlist("files")
    if not files or all(not f.filename for f in files):
        return jsonify({"error": "No files selected. Choose at least one course or faculty document."}), 400

    course_name = clean_text(request.form.get("course_name", "")) or "General Faculty Course"
    department = clean_text(request.form.get("department", "")) or "Academic Affairs"
    try:
        chunk_size = int(request.form.get("chunk_size", 512))
        chunk_overlap = int(request.form.get("chunk_overlap", 96))
    except ValueError:
        chunk_size, chunk_overlap = 512, 96

    use_ollama = request.form.get("use_ollama", "false").lower() == "true"
    ollama_model = clean_text(request.form.get("ollama_model", "phi3:mini")) or "phi3:mini"

    store = load_store()
    existing_chunk_ids = {c["id"] for c in store["chunks"]}
    existing_qa_ids = {q["id"] for q in store["qa_pairs"]}

    added_chunks = 0
    added_qa = 0
    processed_files = []
    errors = []
    engine_used = "offline_heuristic"

    for file_storage in files:
        if not file_storage.filename:
            continue
        filename = _safe_filename(file_storage.filename)
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            errors.append(f"{filename} skipped. Supported formats are PDF, DOCX, PPTX, TXT, MD, CSV, and JSON.")
            continue

        raw = file_storage.read()
        if len(raw) > MAX_FILE_BYTES:
            errors.append(f"{filename} skipped. The file exceeds the 20 MB limit.")
            continue
        if not raw.strip():
            errors.append(f"{filename} skipped. The file is empty.")
            continue

        try:
            sections = parse_uploaded_file(filename, raw)
            if not sections:
                errors.append(f"{filename} produced no readable text. Check if the PDF contains scanned images only.")
                continue

            chunks = chunk_sections(
                sections,
                source_file=filename,
                course_name=course_name,
                department=department,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            qa_pairs, engine_used = extract_qa_pairs(chunks, use_ollama=use_ollama, ollama_model=ollama_model)

            new_chunks = [c for c in chunks if c["id"] not in existing_chunk_ids]
            new_qa = [q for q in qa_pairs if q["id"] not in existing_qa_ids]
            for c in new_chunks:
                existing_chunk_ids.add(c["id"])
            for q in new_qa:
                existing_qa_ids.add(q["id"])

            store["chunks"].extend(new_chunks)
            store["qa_pairs"].extend(new_qa)
            added_chunks += len(new_chunks)
            added_qa += len(new_qa)

            source_entry = {
                "filename": filename,
                "course": course_name,
                "department": department,
                "chunks_count": len(chunks),
                "qa_count": len(qa_pairs),
                "language": detect_language(" ".join(c["text"] for c in chunks)),
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
            }
            store["sources"] = [s for s in store["sources"] if s["filename"] != filename]
            store["sources"].append(source_entry)
            processed_files.append(filename)
        except Exception as exc:
            errors.append(f"{filename} failed to parse. {exc}")

    if not processed_files and errors:
        return jsonify({"error": " ".join(errors)}), 400

    save_store(store)
    return (
        jsonify(
            {
                "message": f"Processed {len(processed_files)} file(s). Gathered {added_chunks} RAG chunks and {added_qa} Q&A pairs.",
                "engine": engine_used,
                "warnings": errors,
                "sources": store["sources"],
                "chunks": store["chunks"],
                "qa_pairs": store["qa_pairs"],
                "stats": _compute_stats(store),
            }
        ),
        200,
    )


@app.route("/api/load-sample", methods=["POST"])
def api_load_sample() -> Response:
    body = request.get_json(silent=True) or {}
    chunk_size = int(body.get("chunk_size", 512))
    chunk_overlap = int(body.get("chunk_overlap", 96))

    chunks, qa_pairs, sources = build_from_sample_documents(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    store = {
        "sources": sources,
        "chunks": chunks,
        "qa_pairs": qa_pairs,
    }
    save_store(store)
    return jsonify(
        {
            "message": f"Loaded {len(sources)} MSA University bilingual files with {len(chunks)} chunks and {len(qa_pairs)} Q&A pairs.",
            "sources": store["sources"],
            "chunks": store["chunks"],
            "qa_pairs": store["qa_pairs"],
            "stats": _compute_stats(store),
        }
    )


@app.route("/api/chunks/<chunk_id>", methods=["PATCH", "DELETE"])
def api_manage_chunk(chunk_id: str) -> tuple[Response, int]:
    store = load_store()
    if request.method == "DELETE":
        before = len(store["chunks"])
        store["chunks"] = [c for c in store["chunks"] if c["id"] != chunk_id]
        if len(store["chunks"]) == before:
            return jsonify({"error": "Chunk not found."}), 404
        save_store(store)
        return jsonify({"message": "Chunk removed from dataset.", "stats": _compute_stats(store)}), 200

    body = request.get_json(silent=True) or {}
    new_text = clean_text(str(body.get("text", "")))
    new_heading = clean_text(str(body.get("heading", "")))
    if not new_text:
        return jsonify({"error": "Chunk text cannot be empty."}), 400

    for chunk in store["chunks"]:
        if chunk["id"] == chunk_id:
            chunk["text"] = new_text
            if new_heading:
                chunk["heading"] = new_heading
            chunk["char_count"] = len(new_text)
            chunk["token_estimate"] = estimate_tokens(new_text)
            chunk["language"] = detect_language(new_text)
            save_store(store)
            return jsonify({"message": "Chunk updated.", "chunk": chunk, "stats": _compute_stats(store)}), 200
    return jsonify({"error": "Chunk not found."}), 404


@app.route("/api/qa", methods=["POST"])
def api_add_qa() -> tuple[Response, int]:
    body = request.get_json(silent=True) or {}
    question = clean_text(str(body.get("question", "")))
    answer = clean_text(str(body.get("answer", "")))
    course = clean_text(str(body.get("course", ""))) or "Manual Entry"
    department = clean_text(str(body.get("department", ""))) or "Student Affairs"
    topic = clean_text(str(body.get("topic", ""))) or "Faculty FAQ"

    if len(question) < 4 or len(answer) < 4:
        return jsonify({"error": "Enter both a question and a grounded answer."}), 400

    store = load_store()
    qa_id = f"qa_manual_{int(datetime.now(timezone.utc).timestamp() * 1000)}"
    record = {
        "id": qa_id,
        "question": question,
        "answer": answer,
        "topic": topic,
        "source_file": "Manual Entry",
        "chunk_id": "manual",
        "course": course,
        "department": department,
        "language": detect_language(f"{question} {answer}"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    store["qa_pairs"].insert(0, record)
    save_store(store)
    return jsonify({"message": "Q&A pair added to dataset.", "qa": record, "stats": _compute_stats(store)}), 201


@app.route("/api/qa/<qa_id>", methods=["PATCH", "DELETE"])
def api_manage_qa(qa_id: str) -> tuple[Response, int]:
    store = load_store()
    if request.method == "DELETE":
        before = len(store["qa_pairs"])
        store["qa_pairs"] = [q for q in store["qa_pairs"] if q["id"] != qa_id]
        if len(store["qa_pairs"]) == before:
            return jsonify({"error": "Q&A pair not found."}), 404
        save_store(store)
        return jsonify({"message": "Q&A pair removed.", "stats": _compute_stats(store)}), 200

    body = request.get_json(silent=True) or {}
    question = clean_text(str(body.get("question", "")))
    answer = clean_text(str(body.get("answer", "")))
    if not question or not answer:
        return jsonify({"error": "Both question and answer are required."}), 400

    for item in store["qa_pairs"]:
        if item["id"] == qa_id:
            item["question"] = question
            item["answer"] = answer
            item["language"] = detect_language(f"{question} {answer}")
            save_store(store)
            return jsonify({"message": "Q&A pair updated.", "qa": item, "stats": _compute_stats(store)}), 200
    return jsonify({"error": "Q&A pair not found."}), 404


@app.route("/api/dataset/clear", methods=["DELETE"])
def api_clear_dataset() -> Response:
    store = _empty_store()
    save_store(store)
    return jsonify({"message": "Dataset workspace cleared.", "stats": _compute_stats(store)})


@app.route("/api/export/<fmt>", methods=["GET"])
def api_export(fmt: str) -> Response:
    store = load_store()
    chunks = store.get("chunks", [])
    qa_pairs = store.get("qa_pairs", [])

    if fmt == "chromadb_json":
        data = json.dumps(build_chromadb_payload(chunks), ensure_ascii=False, indent=2).encode("utf-8")
        return Response(
            data,
            mimetype="application/json; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename=rag_chunks_chromadb.json"},
        )
    if fmt == "jsonl":
        data = export_jsonl_bytes(qa_pairs)
        return Response(
            data,
            mimetype="application/x-ndjson; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename=qa_dataset.jsonl"},
        )
    if fmt == "csv":
        data = export_csv_bytes(chunks, qa_pairs)
        return Response(
            data,
            mimetype="text/csv; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename=campus_dataset.csv"},
        )
    if fmt == "bundle_zip":
        data = export_bundle_zip_bytes(chunks, qa_pairs, INGEST_SCRIPT_PATH)
        return Response(
            data,
            mimetype="application/zip",
            headers={"Content-Disposition": "attachment; filename=campus_assistant_pi_bundle.zip"},
        )
    return jsonify({"error": "Unknown export format."}), 400


@app.route("/api/chroma/ingest", methods=["POST"])
def api_chroma_ingest() -> tuple[Response, int]:
    store = load_store()
    result = ingest_into_local_chroma(store.get("chunks", []), CHROMA_DIR)
    code = 200 if result.get("ok") else 400
    return jsonify(result), code


@app.route("/api/ask", methods=["POST"])
def api_ask() -> tuple[Response, int]:
    body = request.get_json(silent=True) or {}
    question = clean_text(str(body.get("question", "")))
    if not question:
        return jsonify({"error": "Type or speak a student question first."}), 400

    use_ollama = bool(body.get("use_ollama", True))
    ollama_model = clean_text(str(body.get("ollama_model", "")))
    entry_gate = clean_text(str(body.get("entry_gate", "")))
    pending_target = clean_text(str(body.get("pending_target", "")))
    store = load_store()
    result = query_local_rag(
        question=question,
        chunks=store.get("chunks", []),
        qa_pairs=store.get("qa_pairs", []),
        top_k=3,
        use_ollama=use_ollama,
        ollama_model=ollama_model,
        entry_gate=entry_gate,
        pending_target=pending_target,
        rag_engine=RAG_ENGINE,
    )
    return jsonify(result), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    app.run(host="127.0.0.1", port=port, debug=False)
