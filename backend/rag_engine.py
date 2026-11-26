"""Offline Vector RAG Engine using Persistent ChromaDB + Local/Ollama Embeddings & LLM.

Architecture aligned with the Edge AI Avatar Campus Assistant specification:
1. Embedding Layer:
   - Uses Ollama `nomic-embed-text` (768-dim) when local Ollama is running.
   - Falls back to a 100% offline 384-dim L2-normalized bilingual (Arabic + English)
     word + character n-gram dense vector embedder so ChromaDB vector search runs
     with zero internet downloads and handles student typos naturally.
2. Vector Database Layer:
   - Uses Persistent ChromaDB (`data/chroma_db`, cosine HNSW index) whenever chromadb
     is installed, with an automatic pure-Python cosine vector index fallback.
3. Augmented Generation Layer:
   - Injects top-K retrieved vector chunks into a strict grounded prompt for local
     Ollama (`phi3:mini` / `llama3.2:1b` Q4_K_M) when available, or synthesizes
     directly from the retrieved chunks without hallucination.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

TOKEN_RE = re.compile(r"[\u0600-\u06FF]+|[A-Za-z0-9]+")
VECTOR_DIM = 384
COLLECTION_PREFIX = "campus_rag"


def normalize_arabic_token(token: str) -> str:
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


def tokenize_bilingual(text: str) -> List[str]:
    keep_short = {"g", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
    return [
        normalize_arabic_token(tok)
        for tok in TOKEN_RE.findall(text or "")
        if len(tok) > 1 or tok.lower() in keep_short
    ]


def compute_local_dense_embedding(text: str, dim: int = VECTOR_DIM) -> List[float]:
    """Deterministic 384-dim L2-normalized bilingual word + char n-gram vector.

    Runs 100% offline in <1ms on Raspberry Pi 3/4/5 and handles both Arabic/English
    vocabulary and common student spelling typos via character trigrams.
    """
    vec = [0.0] * dim
    tokens = tokenize_bilingual(text)
    if not tokens:
        vec[0] = 1.0
        return vec

    features: List[Tuple[str, float]] = []
    for tok in tokens:
        features.append((f"w:{tok}", 2.2))
        if len(tok) >= 3:
            padded = f"^{tok}$"
            for i in range(len(padded) - 2):
                features.append((f"c3:{padded[i:i+3]}", 0.65))
        if len(tok) >= 4:
            for i in range(len(tok) - 3):
                features.append((f"c4:{tok[i:i+4]}", 0.8))

    for idx in range(len(tokens) - 1):
        features.append((f"bg:{tokens[idx]}_{tokens[idx+1]}", 1.5))

    for feat, weight in features:
        digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        vec[bucket] += sign * weight

    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [round(v / norm, 6) for v in vec]


def list_ollama_models(ollama_url: str = "http://127.0.0.1:11434") -> List[str]:
    """Return all locally installed Ollama model tags."""
    try:
        resp = requests.get(f"{ollama_url}/api/tags", timeout=1.5)
        if resp.status_code == 200:
            return [m.get("name", "") for m in resp.json().get("models", []) if m.get("name")]
    except Exception:
        pass
    return []


def resolve_ollama_model(preferred: str = "", ollama_url: str = "http://127.0.0.1:11434", is_embed: bool = False) -> Optional[str]:
    """Resolve a requested or default Ollama model name to an exact installed tag."""
    models = list_ollama_models(ollama_url)
    if not models:
        return None

    if is_embed:
        for m in models:
            if "nomic-embed-text" in m.lower() or "embed" in m.lower():
                return m
        return None

    chat_models = [m for m in models if "embed" not in m.lower()]
    if not chat_models:
        return None

    if preferred:
        pref_low = preferred.lower().strip()
        for m in chat_models:
            if m.lower() == pref_low:
                return m
        pref_base = pref_low.split(":")[0]
        for m in chat_models:
            if pref_base and pref_base in m.lower():
                return m

    # Default preference: Phi-3 Mini 3.8B Q4_K_M (highest RAG faithfulness), then Llama 3.2 1B, then any chat model
    for hint in ("phi3", "llama3.2", "qwen", "gemma", "mistral"):
        for m in chat_models:
            if hint in m.lower():
                return m
    return chat_models[0]


def _try_ollama_batch_embeddings(
    texts: List[str],
    ollama_url: str = "http://127.0.0.1:11434",
    model: str = "nomic-embed-text:latest",
    batch_size: int = 32,
) -> Optional[List[List[float]]]:
    """Batch-embed texts rapidly via Ollama /api/embed with /api/embeddings fallback."""
    all_vecs: List[List[float]] = []
    safe_texts = [re.sub(r"\s+", " ", (t or " ")).strip()[:1200] or " " for t in texts]
    try:
        for i in range(0, len(safe_texts), batch_size):
            batch = safe_texts[i : i + batch_size]
            resp = requests.post(
                f"{ollama_url}/api/embed",
                json={"model": model, "input": batch, "truncate": True, "keep_alive": "30m"},
                timeout=90,
            )
            if resp.status_code == 200 and isinstance(resp.json().get("embeddings"), list):
                embs = resp.json()["embeddings"]
                if len(embs) == len(batch):
                    all_vecs.extend(embs)
                    continue
            # Fallback to single /api/embeddings if /api/embed unavailable
            for single in batch:
                v = _try_ollama_embedding(single, ollama_url=ollama_url, model=model)
                if v is None:
                    return None
                all_vecs.append(v)
        return all_vecs if len(all_vecs) == len(texts) else None
    except Exception:
        return None


def _try_ollama_embedding(text: str, ollama_url: str = "http://127.0.0.1:11434", model: str = "nomic-embed-text:latest") -> Optional[List[float]]:
    try:
        resp = requests.post(
            f"{ollama_url}/api/embeddings",
            json={"model": model, "prompt": (text or " ")[:1500], "keep_alive": "30m"},
            timeout=15,
        )
        if resp.status_code == 200 and isinstance(resp.json().get("embedding"), list):
            return resp.json()["embedding"]
    except Exception:
        pass
    return None


def _ollama_has_model(model_prefix: str, ollama_url: str = "http://127.0.0.1:11434") -> bool:
    return resolve_ollama_model(model_prefix, ollama_url=ollama_url, is_embed=("embed" in model_prefix.lower())) is not None


class EdgeRAGEngine:
    """Persistent ChromaDB-backed Vector RAG engine with graceful local cosine fallback."""

    def __init__(self, persist_dir: Path) -> None:
        self.persist_dir = persist_dir
        self._client: Any = None
        self._chroma_available = False
        self._in_memory_records: List[Dict[str, Any]] = []
        self._last_embed_mode = "local_bilingual_384d"
        self._indexed_count = 0
        self._init_chroma()

    def _init_chroma(self) -> None:
        try:
            import chromadb  # type: ignore

            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            self._chroma_available = True
            # Detect existing persisted collection count so startup is instant
            for mode in ("ollama_nomic", "local_bilingual_384d"):
                try:
                    col = self._client.get_collection(name=self._collection_name(mode))
                    cnt = col.count()
                    if cnt > 0:
                        self._last_embed_mode = mode
                        self._indexed_count = cnt
                        break
                except Exception:
                    continue
        except Exception:
            self._client = None
            self._chroma_available = False

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "vector_db": "ChromaDB (Persistent)" if self._chroma_available else "Local Cosine Vector Store",
            "chromadb_active": self._chroma_available,
            "embedding_model": (
                "nomic-embed-text (Ollama)" if self._last_embed_mode == "ollama_nomic" else "Bilingual Hybrid Vector (384d)"
            ),
            "indexed_vectors": self._indexed_count,
            "persist_dir": str(self.persist_dir),
        }

    def _collection_name(self, embed_mode: str) -> str:
        return f"{COLLECTION_PREFIX}_{embed_mode}"

    def rebuild_index(self, chunks: List[Dict[str, Any]], qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Embed and upsert all RAG chunks and Q&A pairs into Persistent ChromaDB."""
        records: List[Dict[str, Any]] = []
        for chk in chunks:
            records.append(
                {
                    "id": f"chunk::{chk['id']}",
                    "kind": "chunk",
                    "title": chk.get("heading", ""),
                    "embed_text": f"{chk.get('heading', '')}\n{chk.get('text', '')}",
                    "display_text": chk.get("text", ""),
                    "source_file": chk.get("source_file", "Unknown"),
                    "course": chk.get("course", ""),
                    "language": chk.get("language", "en"),
                }
            )
        for qa in qa_pairs:
            records.append(
                {
                    "id": f"qa::{qa['id']}",
                    "kind": "qa",
                    "title": qa.get("question", ""),
                    "embed_text": f"Q: {qa.get('question', '')}\nA: {qa.get('answer', '')}",
                    "display_text": qa.get("answer", ""),
                    "source_file": qa.get("source_file", "Unknown"),
                    "course": qa.get("course", ""),
                    "language": qa.get("language", "en"),
                }
            )

        embed_model_tag = resolve_ollama_model("nomic-embed-text", is_embed=True) if records else None
        embeddings: List[List[float]] = []
        embed_mode = "local_bilingual_384d"

        if embed_model_tag:
            nomic_vecs = _try_ollama_batch_embeddings(
                [r["embed_text"] for r in records],
                model=embed_model_tag,
                batch_size=32,
            )
            if nomic_vecs:
                embeddings = nomic_vecs
                embed_mode = "ollama_nomic"

        if not embeddings:
            embeddings = [compute_local_dense_embedding(rec["embed_text"]) for rec in records]
            embed_mode = "local_bilingual_384d"

        self._last_embed_mode = embed_mode
        self._indexed_count = len(records)
        self._in_memory_records = [
            {**rec, "embedding": vec} for rec, vec in zip(records, embeddings)
        ]

        if self._chroma_available and self._client is not None:
            col_name = self._collection_name(embed_mode)
            try:
                self._client.delete_collection(name=col_name)
            except Exception:
                pass
            collection = self._client.get_or_create_collection(
                name=col_name,
                metadata={"hnsw:space": "cosine"},
            )
            if records:
                # Upsert in safe batches of 200 to stay under ChromaDB SQLite limits
                for i in range(0, len(records), 200):
                    batch_rec = records[i : i + 200]
                    batch_emb = embeddings[i : i + 200]
                    collection.upsert(
                        ids=[r["id"] for r in batch_rec],
                        documents=[r["display_text"] for r in batch_rec],
                        embeddings=batch_emb,
                        metadatas=[
                            {
                                "kind": r["kind"],
                                "title": r["title"][:240],
                                "source_file": r["source_file"],
                                "course": r["course"],
                                "language": r["language"],
                            }
                            for r in batch_rec
                        ],
                    )

        return self.status

    def retrieve(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        qa_pairs: List[Dict[str, Any]],
        top_k: int = 4,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Retrieve top-K semantically similar records from Persistent ChromaDB (or cosine fallback)."""
        expected_total = len(chunks) + len(qa_pairs)
        if expected_total == 0:
            return [], self.status

        if self._indexed_count != expected_total:
            self.rebuild_index(chunks, qa_pairs)
        elif not self._chroma_available and not self._in_memory_records:
            self.rebuild_index(chunks, qa_pairs)

        query_vec: Optional[List[float]] = None
        if self._last_embed_mode == "ollama_nomic":
            embed_tag = resolve_ollama_model("nomic-embed-text", is_embed=True)
            if embed_tag:
                query_vec = _try_ollama_embedding(question, model=embed_tag)
        if query_vec is None:
            if self._last_embed_mode == "ollama_nomic":
                # Ollama went offline since last index; rebuild local vectors seamlessly
                self.rebuild_index(chunks, qa_pairs)
            query_vec = compute_local_dense_embedding(question)

        results: List[Dict[str, Any]] = []
        n_fetch = min(max(top_k * 4, 16), self._indexed_count)

        if self._chroma_available and self._client is not None:
            try:
                col_name = self._collection_name(self._last_embed_mode)
                collection = self._client.get_collection(name=col_name)
                raw = collection.query(
                    query_embeddings=[query_vec],
                    n_results=n_fetch,
                    include=["documents", "metadatas", "distances"],
                )
                ids = (raw.get("ids") or [[]])[0]
                docs = (raw.get("documents") or [[]])[0]
                metas = (raw.get("metadatas") or [[]])[0]
                dists = (raw.get("distances") or [[]])[0]
                for rec_id, doc_text, meta, dist in zip(ids, docs, metas, dists):
                    sim = max(0.0, min(0.99, 1.0 - float(dist)))
                    results.append(
                        {
                            "id": rec_id.split("::", 1)[-1],
                            "kind": meta.get("kind", "chunk"),
                            "title": meta.get("title", ""),
                            "display_text": doc_text,
                            "source_file": meta.get("source_file", "Unknown"),
                            "course": meta.get("course", ""),
                            "language": meta.get("language", "en"),
                            "vector_similarity": round(sim, 3),
                        }
                    )
            except Exception:
                results = []

        if not results:
            scored: List[Tuple[float, Dict[str, Any]]] = []
            for rec in self._in_memory_records:
                emb = rec["embedding"]
                if len(emb) != len(query_vec):
                    continue
                dot = sum(a * b for a, b in zip(query_vec, emb))
                scored.append((max(0.0, min(0.99, dot)), rec))
            scored.sort(key=lambda pair: pair[0], reverse=True)
            for sim, rec in scored[:n_fetch]:
                results.append(
                    {
                        "id": rec["id"].split("::", 1)[-1],
                        "kind": rec["kind"],
                        "title": rec["title"],
                        "display_text": rec["display_text"],
                        "source_file": rec["source_file"],
                        "course": rec["course"],
                        "language": rec["language"],
                        "vector_similarity": round(sim, 3),
                    }
                )

        # Hybrid rerank: combine ChromaDB cosine similarity + exact bilingual token overlap + Q&A preference
        # Also inject top exact lexical matches (crucial for Arabic queries when using English-heavy nomic-embed-text)
        q_tokens = set(tokenize_bilingual(question))
        seen_keys = {(r["kind"], r["id"]) for r in results}
        if q_tokens:
            lex_candidates: List[Tuple[float, Dict[str, Any]]] = []
            for qa in qa_pairs:
                if ("qa", qa["id"]) in seen_keys:
                    continue
                toks = set(tokenize_bilingual(f"{qa.get('question', '')} {qa.get('answer', '')}"))
                ov = len(q_tokens & toks) / len(q_tokens)
                if ov >= 0.35:
                    lex_candidates.append(
                        (
                            ov + 0.05,
                            {
                                "id": qa["id"],
                                "kind": "qa",
                                "title": qa.get("question", ""),
                                "display_text": qa.get("answer", ""),
                                "source_file": qa.get("source_file", "Unknown"),
                                "course": qa.get("course", ""),
                                "language": qa.get("language", "en"),
                                "vector_similarity": round(min(0.88, 0.55 + ov * 0.35), 3),
                            },
                        )
                    )
            for chk in chunks:
                if ("chunk", chk["id"]) in seen_keys:
                    continue
                toks = set(tokenize_bilingual(f"{chk.get('heading', '')} {chk.get('text', '')}"))
                ov = len(q_tokens & toks) / len(q_tokens)
                if ov >= 0.35:
                    lex_candidates.append(
                        (
                            ov,
                            {
                                "id": chk["id"],
                                "kind": "chunk",
                                "title": chk.get("heading", ""),
                                "display_text": chk.get("text", ""),
                                "source_file": chk.get("source_file", "Unknown"),
                                "course": chk.get("course", ""),
                                "language": chk.get("language", "en"),
                                "vector_similarity": round(min(0.85, 0.52 + ov * 0.33), 3),
                            },
                        )
                    )
            lex_candidates.sort(key=lambda pair: pair[0], reverse=True)
            for _, cand in lex_candidates[:6]:
                results.append(cand)

        for item in results:
            item_tokens = set(tokenize_bilingual(f"{item['title']} {item['display_text']}"))
            overlap = (len(q_tokens & item_tokens) / max(1, len(q_tokens))) if q_tokens else 0.0
            qa_boost = 0.06 if item["kind"] == "qa" else 0.0
            combined = min(0.99, item["vector_similarity"] * 0.72 + overlap * 0.28 + qa_boost)
            item["token_overlap"] = round(overlap, 3)
            item["confidence"] = round(combined, 2)

        results.sort(key=lambda r: r["confidence"], reverse=True)
        return results[:top_k], self.status

    def generate_augmented_answer(
        self,
        question: str,
        retrieved: List[Dict[str, Any]],
        use_ollama: bool = False,
        ollama_model: str = "phi3:mini",
    ) -> Tuple[str, str]:
        """Generate context-aware answer from retrieved vector chunks via Ollama LLM or grounded synthesis."""
        if not retrieved:
            return "NOT_IN_RECORDS", "not_found"

        # Anti-hallucination retrieval relevance gate: reject out-of-scope questions before LLM generation
        top_item = retrieved[0]
        top_sim = float(top_item.get("vector_similarity", 0.0))
        top_overlap = float(top_item.get("token_overlap", 0.0))
        top_conf = float(top_item.get("confidence", 0.0))
        if top_conf < 0.42 or (top_overlap < 0.15 and top_sim < 0.66):
            return "NOT_IN_RECORDS", "not_found"

        context_blocks = "\n\n".join(
            f"[Source {i}: {item['source_file']} · {item['title']}]\n{item['display_text']}"
            for i, item in enumerate(retrieved, start=1)
        )

        # Try local Ollama LLM generation via /api/chat (proper instruct chat template)
        if use_ollama:
            is_ar_q = bool(re.search(r"[\u0600-\u06FF]", question))
            # Prefer multilingual qwen2.5 for Arabic queries when no specific model was forced
            preferred_llm = "qwen2.5" if (is_ar_q and not ollama_model) else ollama_model
            resolved_model = resolve_ollama_model(preferred_llm, is_embed=False)
            if resolved_model:
                system_msg = (
                    "أنت المساعد الجامعي الذكي لجامعة MSA في مدينة 6 أكتوبر. أجب باللغة العربية فقط وباختصار ودقة اعتماداً حصراً على السياق المسترجع أدناه. "
                    "إذا لم يحتوِ السياق المسترجع على إجابة مباشرة للسؤال، أجب بالكلمة التالية فقط: NOT_IN_RECORDS"
                    if is_ar_q
                    else "You are the offline MSA University Campus Assistant in 6th of October City, Egypt. "
                    "Answer ONLY in English, concisely and directly, using strictly the Retrieved Context below. "
                    "If the Retrieved Context does NOT directly contain the answer to the student's question, reply ONLY with the exact token: NOT_IN_RECORDS"
                )
                user_msg = (
                    f"السياق المسترجع:\n{context_blocks}\n\nسؤال الطالب: {question}\nالإجابة:"
                    if is_ar_q
                    else f"Retrieved Context:\n{context_blocks}\n\nStudent Question: {question}\nAnswer:"
                )
                try:
                    resp = requests.post(
                        "http://127.0.0.1:11434/api/chat",
                        json={
                            "model": resolved_model,
                            "messages": [
                                {"role": "system", "content": system_msg},
                                {"role": "user", "content": user_msg},
                            ],
                            "stream": False,
                            "keep_alive": "30m",
                            "options": {"temperature": 0.1, "num_predict": 200, "num_ctx": 2048},
                        },
                        timeout=60,
                    )
                    if resp.status_code == 200:
                        msg_obj = resp.json().get("message") or {}
                        text = (msg_obj.get("content") or "").strip()
                        if "NOT_IN_RECORDS" in text.upper() or re.search(
                            r"(not mentioned in the (?:provided )?context|cannot find (?:this|that) information|no information (?:available|provided) in the context|لا توجد معلومات|غير مذكور في السياق)",
                            text,
                            re.IGNORECASE,
                        ):
                            return "NOT_IN_RECORDS", "not_found"
                        has_ar_out = bool(re.search(r"[\u0600-\u06FF]", text))
                        if text and (is_ar_q == has_ar_out or (not is_ar_q and not has_ar_out)):
                            return text, f"chromadb_rag+ollama:{resolved_model}"
                except Exception:
                    pass

        # Grounded extractive RAG synthesis directly from top ChromaDB chunks
        primary = retrieved[0]["display_text"]
        answer = primary
        if len(retrieved) > 1:
            second = retrieved[1]
            if (
                second["confidence"] >= retrieved[0]["confidence"] * 0.80
                and second["display_text"] not in primary
                and primary not in second["display_text"]
                and len(primary) + len(second["display_text"]) < 1200
            ):
                answer = f"{primary}\n\n{second['display_text']}"

        gen_mode = "chromadb_vector_rag" if self._chroma_available else "local_vector_rag"
        return answer, gen_mode
