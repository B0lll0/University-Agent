# Edge AI Avatar Campus Assistant — MSA University

Offline Edge AI-powered bilingual (Arabic & English) campus assistant for new student onboarding at **MSA University (October University for Modern Sciences and Arts, 6th of October City, Egypt)**, built with Python Flask, Persistent ChromaDB, and local Ollama (`Q4_K_M` quantized LLMs + `nomic-embed-text`).

## Interfaces

- **Dedicated Edge AI Kiosk (`/edge` & `/kiosk`):** Clean touch-kiosk UI with a top-centered animated SVG Avatar, full-window conversation, zero developer dropdowns, and locked to **`qwen2.5:1.5b-instruct-q4_K_M`** (~940 MB, bilingual Arabic + English) for Raspberry Pi 4 deployment.
- **Full Configurable Studio (`/`):** Animated Avatar + chat window with selectable local Ollama models (`phi3:3.8b`, `qwen2.5:1.5b`, `llama3.2:1b`, or instant RAG-only), speech language, TTS output engine, and speed controls.

## Knowledge Base & Interactive Flows

- **54 Indexed Sources (826 RAG chunks, 245 Q&A pairs, 1,071 ChromaDB vectors):**
  - 10 Official MSA University bilingual guides: 2026/2027 Egyptian (EGP) & International (USD) tuition categories (A/B/C/D) for all 11 faculties, UK validation fees (Greenwich £440 / Bedfordshire £335), scholarships & parent discounts, bus fees, minimum admission scores, ELPT rules & exemptions, transfer rules, required admission documents, campus buildings & Gate 1–4 directions, food court, and full 4/5-year study plans across all 27 majors.
  - 44 course & project PDFs (Artificial Intelligence & Microprocessors lectures, labs, tutorials, and project proposal).
- **Interactive Campus Flows:**
  - **Room & Gate Locator:** Ground-level (`1xx`) vs. upper-floor directions from Gate 1–4.
  - **Faculties → Majors → Study Plan → Faculty-Aware Q1–Q5:** Tailored scholarships (Q4), tuition fees (Q3), dual degrees (Q1), syndicates (Q2), and transfer GPA (Q5) for the selected faculty.
  - **Tuition Categories Advisor:** All 11 faculties' exact 2026/2027 Categories A–D in EGP & USD.
  - **3 Context-Aware Follow-Up Questions** after every answer + **Anti-Hallucination Guardrail** directing out-of-scope questions to Student Affairs (SSB) with interactive gate wayfinding.

## Edge Footprint (Raspberry Pi 4)

- **Active Parameters (`/edge`):** ~1.64B (`qwen2.5:1.5b-instruct-q4_K_M` + `nomic-embed-text` 137M)
- **Disk Footprint:** ~1.69 GB total (models + Python env + ChromaDB + app)
- **Runtime RAM:** ~1.8 – 2.2 GB RAM (fits comfortably on 4 GB / 8 GB Raspberry Pi 4)

## Quick Start (Windows / Linux / macOS)

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# Pull local Ollama models
ollama pull nomic-embed-text
ollama pull qwen2.5:1.5b-instruct-q4_K_M

python app.py
```

Open `http://127.0.0.1:5050/edge` (Edge Kiosk) or `http://127.0.0.1:5050/` (Full Studio).

## Turnkey Raspberry Pi 4/5 Deployment

```bash
chmod +x scripts/deploy_raspberry_pi.sh scripts/kiosk_start.sh
./scripts/deploy_raspberry_pi.sh
./scripts/kiosk_start.sh
```
