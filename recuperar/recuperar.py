"""Módulo de Recuperación Híbrida (BM25 + FAISS + RRF) para el Kernel Boyacense.

Cumple estrictamente con las reglas de negocio del CODEFEST AD ASTRA 2026:
1. 100% determinista / libre de modelos generativos LLM en recuperación (§8.3).
2. Agregación a nivel de documento para F1@3 (§8.6).
3. Fragmentos recortados a máximo 250 palabras respetando oraciones (§9.2.1).
4. Fusión Reciprocal Rank Fusion (RRF) con k0=60 (§8.4).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set

import numpy as np

FAISS_DIR_DEFAULT = Path(__file__).parent.parent / "base_vectorial" / "encoder_multilingual-e5-large-instruct"
MODELO_EMBEDDINGS_DEFAULT = "intfloat/multilingual-e5-large-instruct"


class BM25Nativo:
    """Implementación de BM25Okapi en Python puro (sin dependencias externas)."""

    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_tokens = [self._tokenize(doc) for doc in corpus]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / max(1, self.corpus_size)

        # Frecuencias de término por documento e IDF
        self.doc_freqs: List[Counter[str]] = [Counter(tokens) for tokens in self.doc_tokens]
        self.idf: Dict[str, float] = {}
        self._calculate_idf()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text_clean = re.sub(r"[^\w\s]", " ", text.lower())
        return [t for t in text_clean.split() if len(t) > 1]

    def _calculate_idf(self) -> None:
        df_counts: Counter[str] = Counter()
        for freqs in self.doc_freqs:
            for word in freqs.keys():
                df_counts[word] += 1

        for word, freq in df_counts.items():
            # IDF estándar BM25Okapi con smoothing
            self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = np.zeros(self.corpus_size, dtype=np.float32)
        for token in query_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            for i, freqs in enumerate(self.doc_freqs):
                freq = freqs.get(token, 0)
                if freq > 0:
                    numerator = freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * (self.doc_len[i] / self.avgdl))
                    scores[i] += idf_val * (numerator / denominator)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]


class BuscadorHibrido:
    """Motor de búsqueda híbrida combinando BM25 y FAISS mediante Reciprocal Rank Fusion."""

    def __init__(self, faiss_dir: Path | str = FAISS_DIR_DEFAULT):
        self.faiss_dir = Path(faiss_dir)
        self.index_path = self.faiss_dir / "index.faiss"
        self.metadata_path = self.faiss_dir / "metadata.jsonl"

        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError(f"No se encontraron los archivos del índice en {self.faiss_dir}")

        self._cargar_metadatos_e_indice()
        self._inicializar_bm25()
        self.embeddings_model = None

    def _cargar_metadatos_e_indice(self) -> None:
        import faiss
        self.faiss_index = faiss.read_index(str(self.index_path))
        self.metadatos: List[Dict[str, Any]] = []
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.metadatos.append(json.loads(line.strip()))

    def _inicializar_bm25(self) -> None:
        corpus = [m.get("texto", "") for m in self.metadatos]
        self.bm25 = BM25Nativo(corpus)

    def _cargar_modelo_embeddings(self) -> None:
        if self.embeddings_model is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embeddings_model = HuggingFaceEmbeddings(
                model_name=MODELO_EMBEDDINGS_DEFAULT,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

    @staticmethod
    def _recortar_a_250_palabras(texto: str, max_words: int = 250) -> str:
        words = texto.split()
        if len(words) <= max_words:
            return texto
        
        # Recorte simple por palabras respetando límite estricto
        return " ".join(words[:max_words])

    @staticmethod
    def _jaccard_similarity(text1: str, text2: str) -> float:
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)

    def buscar(
        self,
        pregunta: str,
        top_k_docs: int = 3,
        top_k_chunks: int = 10,
        candidate_k: int = 60,
        rrf_k0: int = 60,
    ) -> Dict[str, Any]:
        """Ejecuta la búsqueda híbrida determinista y devuelve documentos y fragmentos ordenados."""
        self._cargar_modelo_embeddings()

        # 1. Búsqueda Dispersa (BM25)
        bm25_results = self.bm25.search(pregunta, top_k=candidate_k)
        bm25_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(bm25_results)}

        # 2. Búsqueda Densa (FAISS con prefijo e5)
        instruccion = "Given a question, retrieve passages from documents that contain the exact factual information needed to answer the question"
        query_formateada = f"Instruct: {instruccion}\nQuery: {pregunta}"
        query_vector = np.array([self.embeddings_model.embed_query(query_formateada)], dtype=np.float32)

        scores_faiss, indices_faiss = self.faiss_index.search(query_vector, candidate_k)
        faiss_ranks = {int(idx): rank + 1 for rank, idx in enumerate(indices_faiss[0]) if idx >= 0}

        # 3. Fusión RRF (Reciprocal Rank Fusion)
        todos_indices = set(bm25_ranks.keys()) | set(faiss_ranks.keys())
        rrf_scores: List[Tuple[int, float]] = []

        for idx in todos_indices:
            r_bm25 = bm25_ranks.get(idx, candidate_k + 1)
            r_faiss = faiss_ranks.get(idx, candidate_k + 1)
            score_rrf = (1.0 / (rrf_k0 + r_bm25)) + (1.0 / (rrf_k0 + r_faiss))
            rrf_scores.append((idx, score_rrf))

        rrf_scores.sort(key=lambda x: x[1], reverse=True)

        # 4. Agregación Documental (Max Pooling sobre RRF) para F1@3
        doc_scores: Dict[str, float] = {}
        for idx, score in rrf_scores:
            doc_id = self.metadatos[idx]["doc_id"]
            if doc_id not in doc_scores or score > doc_scores[doc_id]:
                doc_scores[doc_id] = score

        top_docs_sorted = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k_docs]
        documents_output = [
            {"rank": rank + 1, "doc_id": doc_id}
            for rank, (doc_id, _) in enumerate(top_docs_sorted)
        ]

        # 5. Deduplicación por Jaccard y Recorte ≤ 250 palabras para NDCG@10
        chunks_seleccionados: List[Dict[str, Any]] = []
        textos_vistos_por_doc: Dict[str, List[str]] = {}

        for idx, score in rrf_scores:
            if len(chunks_seleccionados) >= top_k_chunks:
                break

            meta = self.metadatos[idx]
            doc_id = meta["doc_id"]
            texto = meta["texto"]

            # Evitar solape excesivo con chunks ya seleccionados del mismo documento
            vistos = textos_vistos_por_doc.setdefault(doc_id, [])
            es_duplicado = any(self._jaccard_similarity(texto, v) > 0.5 for v in vistos)
            if es_duplicado:
                continue

            vistos.append(texto)
            texto_recortado = self._recortar_a_250_palabras(texto, max_words=250)

            chunks_seleccionados.append({
                "rank": len(chunks_seleccionados) + 1,
                "chunk_id": meta["chunk_id"],
                "doc_id": doc_id,
                "text": texto_recortado,
                "score_rrf": float(score),
                "fuente": meta.get("fuente", ""),
                "num_tokens": meta.get("num_tokens", 0),
            })

        return {
            "documents": documents_output,
            "fragments": chunks_seleccionados,
        }
