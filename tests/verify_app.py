"""End-to-end verification of Flask API, parsers, exports, and SKILL.md visual/copy rules."""

from __future__ import annotations

import io
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import app  # noqa: E402


def verify_skill_rules() -> None:
    css = (ROOT / "static" / "css" / "design-system.css").read_text(encoding="utf-8")
    i18n = (ROOT / "static" / "js" / "i18n.js").read_text(encoding="utf-8")

    # 1. No italic font-style in CSS
    assert "font-style: italic" not in css, "Found forbidden italic font style in CSS"

    # 2. No background gradients (only linear-gradient allowed is .hero-headline text gradient)
    grad_matches = re.findall(r"linear-gradient\([^)]+\)", css)
    assert len(grad_matches) == 1, f"Expected exactly 1 gradient (hero text), found {len(grad_matches)}"

    # 3. No single-sided card borders
    single_side = re.findall(r"border-(?:top|bottom|left|right)\s*:", css)
    assert not single_side, f"Found single-sided borders in CSS: {single_side}"

    # 4. No banned fonts in font-family declarations
    font_decls = re.findall(r"font-family\s*:[^;]+;", css)
    for decl in font_decls:
        for bad_font in ("Inter", "Roboto", "Arial", "Open Sans", "Helvetica"):
            assert not re.search(rf"\b{re.escape(bad_font)}\b", decl), f"Found banned font {bad_font} in {decl}"

    # 5. Zero hyphens inside i18n visible strings & zero banned filler words
    banned_words = [
        "revolutionize",
        "supercharge",
        "cutting edge",
        "effortless",
        "transform your workflow",
        "take it to the next level",
        "harness the power",
        "world class",
    ]
    string_literals = re.findall(r":\s*'([^']*)'", i18n)
    for lit in string_literals:
        assert "-" not in lit, f"Found forbidden hyphen in visible i18n copy: {lit!r}"
        assert "!" not in lit, f"Found exclamation mark in i18n copy: {lit!r}"
        low = lit.lower()
        for bw in banned_words:
            assert bw not in low, f"Found banned filler phrase {bw!r} in {lit!r}"

    print("[PASS] SKILL.md design system & copy checks passed.")


def verify_flask_api() -> None:
    client = app.test_client()

    # Pages & 404
    assert client.get("/").status_code == 200
    assert client.get("/kiosk").status_code == 200
    assert client.get("/privacy").status_code == 200
    assert client.get("/terms").status_code == 200
    assert client.get("/missing-route-check").status_code == 404

    # Health, Live Persistent ChromaDB RAG, & Offline TTS endpoint
    health = client.get("/api/health").get_json()
    assert health["status"] == "ok" and health["offline_ready"] is True
    assert "tts" in health
    assert health["rag"]["chromadb_active"] is True, health["rag"]
    assert "ChromaDB" in health["rag"]["vector_db"], health["rag"]
    tts_resp = client.post("/api/tts", json={"text": "Welcome to the campus assistant.", "lang": "en", "rate": 1.0})
    assert tts_resp.status_code in (200, 503)
    if tts_resp.status_code == 200:
        assert tts_resp.data[:4] == b"RIFF"

    # Clear first
    assert client.delete("/api/dataset/clear").status_code == 200

    # Load bilingual MSA University dataset (7 markdown files + full 11-faculty study plans)
    sample_resp = client.post("/api/load-sample", json={"chunk_size": 768, "chunk_overlap": 96})
    assert sample_resp.status_code == 200
    sample_data = sample_resp.get_json()
    assert sample_data["stats"]["sources_count"] == 8
    assert sample_data["stats"]["chunks_count"] >= 38
    assert sample_data["stats"]["qa_count"] >= 42

    # Verify anti-hallucination refusal + interactive Student Affairs guidance flow (EN & AR)
    unknown_en = client.post("/api/ask", json={"question": "Who won the 2022 FIFA World Cup in Qatar?"}).get_json()
    assert "I don't have that information in my faculty records." in unknown_en["answer"], unknown_en
    assert "Please check with Student Affairs." in unknown_en["answer"], unknown_en
    assert "Would you like me to guide you to the Student Affairs location?" in unknown_en["answer"], unknown_en
    assert unknown_en["pending_target"] == "GUIDE_STUDENT_AFFAIRS" and len(unknown_en["gate_options"]) == 2, unknown_en

    guide_yes = client.post("/api/ask", json={"question": "Yes", "pending_target": "GUIDE_STUDENT_AFFAIRS"}).get_json()
    assert "Student Services Building" in guide_yes["answer"] and guide_yes["pending_target"] == "SSB", guide_yes
    assert [o["gate"] for o in guide_yes["gate_options"]] == ["1", "2", "3", "4"], guide_yes

    guide_g2 = client.post("/api/ask", json={"question": "Gate 2", "entry_gate": "2", "pending_target": "SSB"}).get_json()
    assert guide_g2["answer"].startswith("Directions from Gate 2:"), guide_g2

    unknown_ar = client.post("/api/ask", json={"question": "من فاز بكأس العالم لكرة القدم في قطر؟"}).get_json()
    assert "يرجى مراجعة شؤون الطلاب" in unknown_ar["answer"] and unknown_ar["pending_target"] == "GUIDE_STUDENT_AFFAIRS", unknown_ar

    # Verify Food Court & Restaurants query returns direct bulleted restaurant list + 3 contextual follow-up Qs (EN & AR)
    fc_en = client.post("/api/ask", json={"question": "What restaurants are in the food court?"}).get_json()
    assert "• Qadra" in fc_en["answer"] and "• Krispy Kreme" in fc_en["answer"] and "• Cinnabon" in fc_en["answer"], fc_en
    assert len(fc_en["suggested_questions"]) == 3, fc_en
    fc_ar = client.post("/api/ask", json={"question": "إيه المطاعم في الفود كورت؟"}).get_json()
    assert "• قدرة" in fc_ar["answer"] and "• كريسبي كريم" in fc_ar["answer"], fc_ar
    assert len(fc_ar["suggested_questions"]) == 3, fc_ar

    # Verify 3-step bulleted MSA Faculties -> Majors -> Full 5-Year / 4-Year Study Plan flow (EN & AR)
    fac_q = client.post("/api/ask", json={"question": "what are the fucalties the univeersty have"}).get_json()
    assert "• Engineering" in fac_q["answer"] and "• Biotechnology" in fac_q["answer"] and "• Mass Communication" in fac_q["answer"], fac_q
    assert "Which faculty are you interested in?" in fac_q["answer"], fac_q
    assert fac_q["needs_gate"] is True and fac_q["pending_target"] == "MSA_FACULTY_SELECT", fac_q
    assert len(fac_q["gate_options"]) == 11, fac_q

    # Step 2: Select Engineering -> bulleted majors + "Which major are you interested in?"
    eng_majors = client.post("/api/ask", json={"question": "Engineering", "pending_target": "MSA_FACULTY_SELECT"}).get_json()
    assert "• Architectural Systems Engineering (ASE)" in eng_majors["answer"] and "• Computer Systems Engineering (CSE)" in eng_majors["answer"], eng_majors
    assert "Which major are you interested in?" in eng_majors["answer"], eng_majors
    assert eng_majors["pending_target"] == "MSA_MAJOR_SELECT:engineering", eng_majors
    assert len(eng_majors["gate_options"]) == 5, eng_majors

    # Step 3: Select CSE -> full Five Year Study Plan bullet points + Top 5 Frequently Asked Questions
    cse_plan = client.post("/api/ask", json={"question": "Computer Systems Engineering (CSE)", "pending_target": "MSA_MAJOR_SELECT:engineering"}).get_json()
    for yr in ("• Year 1", "• Year 2", "• Year 3", "• Year 4", "• Year 5"):
        assert yr in cse_plan["answer"], cse_plan
    assert "Top 5 frequently asked questions" in cse_plan["answer"], cse_plan
    assert len(cse_plan["gate_options"]) == 5, cse_plan
    assert cse_plan["needs_gate"] is False, cse_plan

    # Clicking one of the Top 5 FAQs returns a real ChromaDB RAG answer
    faq_reply = client.post("/api/ask", json={"question": cse_plan["gate_options"][1]["query"]}).get_json()
    assert "Syndicate" in faq_reply["answer"] and faq_reply["mode"] == "chromadb_vector_rag", faq_reply

    # Test another faculty: Mass Communication -> Broadcast 4-year plan
    mc_majors = client.post("/api/ask", json={"question": "mass com", "pending_target": "MSA_FACULTY_SELECT"}).get_json()
    assert "Which major are you interested in?" in mc_majors["answer"] and "• Broadcast Platforms" in mc_majors["answer"], mc_majors
    mc_plan = client.post("/api/ask", json={"question": "Broadcast Platforms (TV, Radio & Webcasting)", "pending_target": "MSA_MAJOR_SELECT:mass_communication"}).get_json()
    assert "• Year 1" in mc_plan["answer"] and "• Year 4" in mc_plan["answer"], mc_plan

    # Arabic 3-step flow: Faculties -> Engineering -> Mechatronics 5-year plan
    fac_ar = client.post("/api/ask", json={"question": "ما هي كليات جامعة MSA؟"}).get_json()
    assert "• الهندسة" in fac_ar["answer"] and "ما هي الكلية التي تهتم بها؟" in fac_ar["answer"], fac_ar
    eng_ar = client.post("/api/ask", json={"question": "الهندسة", "pending_target": "MSA_FACULTY_SELECT"}).get_json()
    assert "• هندسة نظم الميكاترونكس (MSE)" in eng_ar["answer"] and "ما هو التخصص الذي تهتم به" in eng_ar["answer"], eng_ar
    mse_ar = client.post("/api/ask", json={"question": "الميكاترونكس", "pending_target": "MSA_MAJOR_SELECT:engineering"}).get_json()
    assert "• السنة الأولى" in mse_ar["answer"] and "• السنة الخامسة" in mse_ar["answer"], mse_ar

    # Verify two-step interactive gate flow for MSA room queries (EN & AR)
    r_c211 = client.post("/api/ask", json={"question": "Where is room C211?"}).get_json()
    assert r_c211["answer"].startswith("Room C211 is located in Building C (Building 3), Second floor, Room 11."), r_c211
    assert r_c211["needs_gate"] is True and r_c211["pending_target"] == "C211", r_c211
    assert [o["gate"] for o in r_c211["gate_options"]] == ["1", "2", "3", "4"], r_c211
    assert "Main scientific academic complex" not in r_c211["answer"], r_c211
    assert "Gate 1:" not in r_c211["answer"], r_c211

    # Step 2: student clicks Gate 2 after C211 -> ONLY the directions paragraph (no repeated header)
    r_c211_g2 = client.post("/api/ask", json={"question": "Gate 2", "entry_gate": "2", "pending_target": "C211"}).get_json()
    assert r_c211_g2["answer"].startswith("Directions from Gate 2:"), r_c211_g2
    assert "Room C211 is located in" not in r_c211_g2["answer"], r_c211_g2
    assert "stairs or elevator" in r_c211_g2["answer"], r_c211_g2
    assert r_c211_g2["needs_gate"] is False and r_c211_g2["gate_options"] == [], r_c211_g2

    # First floor (1xx like C111) is ground level (no stairs/elevator needed)
    r_c111_g1 = client.post("/api/ask", json={"question": "Gate 1", "entry_gate": "1", "pending_target": "C111"}).get_json()
    assert r_c111_g1["answer"].startswith("Directions from Gate 1:"), r_c111_g1
    assert "ground level" in r_c111_g1["answer"] and "no stairs or elevator needed" in r_c111_g1["answer"], r_c111_g1
    assert "Room C111 is located in" not in r_c111_g1["answer"], r_c111_g1

    # Direct single-turn with gate already mentioned
    r_k108_g4 = client.post("/api/ask", json={"question": "Where is room K108 from gate 4?"}).get_json()
    assert r_k108_g4["answer"].startswith("Room K108 is located in Building K, First floor, Room 08."), r_k108_g4
    assert r_k108_g4["needs_gate"] is False and "Directions from Gate 4:" in r_k108_g4["answer"], r_k108_g4

    # Arabic two-step interactive flow -> follow-up returns ONLY directions paragraph
    r_b205_ar_ask = client.post("/api/ask", json={"question": "فين قاعة B205؟"}).get_json()
    assert r_b205_ar_ask["needs_gate"] is True and r_b205_ar_ask["pending_target"] == "B205", r_b205_ar_ask
    assert [o["label"] for o in r_b205_ar_ask["gate_options"]] == ["بوابة 1", "بوابة 2", "بوابة 3", "بوابة 4"], r_b205_ar_ask

    r_b205_ar_g2 = client.post("/api/ask", json={"question": "بوابة 2", "pending_target": "B205"}).get_json()
    assert r_b205_ar_g2["answer"].startswith("الاتجاهات من بوابة 2:"), r_b205_ar_g2
    assert "قاعة B205 موجودة" not in r_b205_ar_g2["answer"] and r_b205_ar_g2["needs_gate"] is False, r_b205_ar_g2

    r_lib = client.post("/api/ask", json={"question": "Where is the library?"}).get_json()
    assert "Building 7" in r_lib["answer"], r_lib

    r_cs_ar = client.post("/api/ask", json={"question": "فين كلية علوم الحاسب؟"}).get_json()
    assert "مبنى 3" in r_cs_ar["answer"], r_cs_ar

    msa_arch = client.post("/api/ask", json={"question": "What is the architectural design and campus layout of MSA University in 6th of October?"}).get_json()
    assert "50 acre" in msa_arch["answer"] and "Opera House" in msa_arch["answer"], msa_arch

    msa_bldg = client.post("/api/ask", json={"question": "What is located in Building 1, Building 2 (SSB), and Building 3 at MSA University?"}).get_json()
    assert "Student Services Building" in msa_bldg["answer"], msa_bldg

    msa_rules = client.post("/api/ask", json={"question": "What is the attendance rule and absence limit at MSA University?"}).get_json()
    assert "75 percent" in msa_rules["answer"], msa_rules
    assert msa_rules["mode"] == "chromadb_vector_rag", msa_rules
    assert "ChromaDB" in msa_rules["rag_trace"]["vector_db"], msa_rules["rag_trace"]
    assert len(msa_rules["rag_trace"]["retrieved_chunks"]) >= 1, msa_rules["rag_trace"]

    msa_ar = client.post("/api/ask", json={"question": "ما هو دليل مباني جامعة MSA ومبنى G؟"}).get_json()
    assert "مبنى" in msa_ar["answer"] and ("1150" in msa_ar["answer"] or "SSB" in msa_ar["answer"]), msa_ar

    # Upload custom TXT + CSV files
    txt_bytes = (
        "# Robotics and Edge Sensors\n"
        "Q: Which microcontroller connects remote kiosks to the Raspberry Pi?\n"
        "A: ESP32 S3 modules connect distributed info kiosks around campus via MQTT.\n"
    ).encode("utf-8")
    csv_bytes = (
        "Question,Answer\n"
        "متى تبدأ المحاضرات الصباحية؟,تبدأ المحاضرات الساعة التاسعة صباحاً في مبنى 3.\n"
    ).encode("utf-8")

    up_resp = client.post(
        "/api/upload",
        data={
            "course_name": "CS201 Embedded AI",
            "department": "Computer Science & AI",
            "chunk_size": "512",
            "chunk_overlap": "96",
            "files": [
                (io.BytesIO(txt_bytes), "edge_sensors.txt"),
                (io.BytesIO(csv_bytes), "arabic_schedule.csv"),
            ],
        },
        content_type="multipart/form-data",
    )
    assert up_resp.status_code == 200, up_resp.get_data(as_text=True)
    up_data = up_resp.get_json()
    assert up_data["stats"]["sources_count"] == 10

    # Ask in English
    ask_en = client.post("/api/ask", json={"question": "Which microcontroller connects remote kiosks?"}).get_json()
    assert "ESP32 S3" in ask_en["answer"]
    assert ask_en["latency_ms"] < 2000

    # Ask in Arabic
    ask_ar = client.post("/api/ask", json={"question": "ما هي قواعد الحضور والغياب في جامعة MSA؟"}).get_json()
    assert "75%" in ask_ar["answer"]
    assert ask_ar["latency_ms"] < 2000

    # Manual Q&A add, edit, delete
    add_qa = client.post("/api/qa", json={"question": "Where is the library?", "answer": "Building D second floor."})
    assert add_qa.status_code == 201
    qa_id = add_qa.get_json()["qa"]["id"]
    assert client.patch(f"/api/qa/{qa_id}", json={"question": "Where is the library?", "answer": "Building D, 2nd floor."}).status_code == 200
    assert client.delete(f"/api/qa/{qa_id}").status_code == 200

    # Exports
    chroma_resp = client.get("/api/export/chromadb_json")
    assert chroma_resp.status_code == 200
    chroma_json = json.loads(chroma_resp.data.decode("utf-8"))
    assert chroma_json["embedding_model"] == "nomic-embed-text"
    assert len(chroma_json["documents"]) == len(chroma_json["ids"]) > 0

    jsonl_resp = client.get("/api/export/jsonl")
    assert jsonl_resp.status_code == 200
    lines = [ln for ln in jsonl_resp.data.decode("utf-8").splitlines() if ln.strip()]
    assert len(lines) > 0 and "instruction" in json.loads(lines[0])

    csv_resp = client.get("/api/export/csv")
    assert csv_resp.status_code == 200
    assert csv_resp.data.startswith(b"\xef\xbb\xbf")  # UTF-8 BOM for Arabic Excel

    zip_resp = client.get("/api/export/bundle_zip")
    assert zip_resp.status_code == 200
    with zipfile.ZipFile(io.BytesIO(zip_resp.data)) as zf:
        names = set(zf.namelist())
        assert {
            "rag_chunks_chromadb.json",
            "qa_dataset.jsonl",
            "campus_dataset.csv",
            "ingest_chromadb.py",
            "deploy_raspberry_pi.sh",
            "kiosk_start.sh",
        } <= names

    # Leave MSA University bilingual dataset loaded cleanly in data/dataset_store.json
    client.post("/api/load-sample", json={"chunk_size": 768, "chunk_overlap": 96})
    print("[PASS] All Flask routes, parsers, Arabic/English RAG queries, and exports verified.")


if __name__ == "__main__":
    verify_skill_rules()
    verify_flask_api()
