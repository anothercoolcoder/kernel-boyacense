"""Módulo de Recuperación Híbrida (BM25 + FAISS + CombSUM) para el Kernel Boyacense.

Cumple estrictamente con las reglas de negocio del CODEFEST AD ASTRA 2026:
1. 100% determinista / libre de modelos generativos LLM en recuperación (§8.3).
2. Agregación a nivel de documento para F1@3 (§8.6).
3. Fragmentos recortados a máximo 250 palabras respetando oraciones (§9.2.1).
4. Fusión CombSUM con RRF disponible como baseline (§8.4).
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
    """Motor de búsqueda híbrida combinando BM25 y FAISS."""

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
        if contar_palabras(texto) <= max_words:
            return texto

        # Acumula oraciones completas para no cortar evidencia lingüística.
        oraciones = re.split(r"(?<=[.!?])\s+", texto.strip())
        resultado: List[str] = []
        palabras = 0
        for oracion in oraciones:
            cantidad = contar_palabras(oracion)
            if not oracion or palabras + cantidad > max_words:
                break
            resultado.append(oracion)
            palabras += cantidad

        if resultado:
            return " ".join(resultado)

        # Caso extremo: una sola oración supera el contrato; prioriza límite
        # formal para evitar emitir fragmentos inválidos.
        return " ".join(texto.split()[:max_words])

    @staticmethod
    def _jaccard_similarity(text1: str, text2: str) -> float:
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)

    @staticmethod
    def _normalizar_scores(
        scores: Dict[int, float], method: str = "minmax"
    ) -> Dict[int, float]:
        """Normaliza scores a [0, 1] con método experimental explícito."""
        if not scores:
            return {}
        if method == "percentile":
            valores = sorted(scores.values())
            divisor = max(1, len(valores) - 1)
            return {
                idx: sum(value < score for value in valores) / divisor
                for idx, score in scores.items()
            }
        if method == "zsigmoid":
            valores = np.array(list(scores.values()), dtype=np.float64)
            desviacion = float(np.std(valores))
            if desviacion == 0.0:
                return {idx: 1.0 for idx in scores}
            media = float(np.mean(valores))
            return {
                idx: 1.0 / (1.0 + math.exp(-((score - media) / desviacion)))
                for idx, score in scores.items()
            }
        if method not in {"minmax", "clipped_minmax"}:
            raise ValueError("normalization_method inválido")
        valores = np.array(list(scores.values()), dtype=np.float64)
        if method == "clipped_minmax":
            minimo = float(np.percentile(valores, 5))
            maximo = float(np.percentile(valores, 95))
        else:
            minimo = float(np.min(valores))
            maximo = float(np.max(valores))
        if maximo == minimo:
            return {idx: 1.0 for idx in scores}
        return {idx: float(np.clip((score - minimo) / (maximo - minimo), 0.0, 1.0)) for idx, score in scores.items()}

    def buscar(
        self,
        pregunta: str,
        top_k_docs: int = 3,
        top_k_chunks: int = 10,
        candidate_k: int = 100,
        rrf_k0: int = 60,
        fusion_method: str = "combsum",
        normalization_method: str = "minmax",
        alpha: float = 0.4,
    ) -> Dict[str, Any]:
        """Ejecuta la búsqueda híbrida determinista y devuelve documentos y fragmentos ordenados."""
        if fusion_method not in {"combsum", "rrf"}:
            raise ValueError("fusion_method debe ser 'combsum' o 'rrf'")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha debe estar entre 0.0 y 1.0")
        self._cargar_modelo_embeddings()

        # 1. Búsqueda Dispersa (BM25)
        bm25_results = self.bm25.search(pregunta, top_k=candidate_k)
        bm25_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(bm25_results)}
        bm25_scores = {idx: score for idx, score in bm25_results}

        # 2. Búsqueda Densa (FAISS con prefijo e5)
        instruccion = "Given a question, retrieve passages from documents that contain the exact factual information needed to answer the question"
        query_formateada = f"Instruct: {instruccion}\nQuery: {pregunta}"
        query_vector = np.array([self.embeddings_model.embed_query(query_formateada)], dtype=np.float32)

        scores_faiss, indices_faiss = self.faiss_index.search(query_vector, candidate_k)
        faiss_ranks = {int(idx): rank + 1 for rank, idx in enumerate(indices_faiss[0]) if idx >= 0}
        faiss_scores = {
            int(idx): float(score)
            for score, idx in zip(scores_faiss[0], indices_faiss[0])
            if idx >= 0
        }

        # 3. Fusión configurable: CombSUM principal, RRF para baseline.
        todos_indices = set(bm25_ranks.keys()) | set(faiss_ranks.keys())
        fused_scores: List[Tuple[int, float]] = []

        if fusion_method == "combsum":
            bm25_normalizados = self._normalizar_scores(bm25_scores, normalization_method)
            faiss_normalizados = self._normalizar_scores(faiss_scores, normalization_method)
            for idx in todos_indices:
                score = (
                    alpha * bm25_normalizados.get(idx, 0.0)
                    + (1.0 - alpha) * faiss_normalizados.get(idx, 0.0)
                )
                fused_scores.append((idx, score))
        else:
            bm25_normalizados = {}
            faiss_normalizados = {}
            for idx in todos_indices:
                r_bm25 = bm25_ranks.get(idx, candidate_k + 1)
                r_faiss = faiss_ranks.get(idx, candidate_k + 1)
                score = (1.0 / (rrf_k0 + r_bm25)) + (1.0 / (rrf_k0 + r_faiss))
                fused_scores.append((idx, score))

        fused_scores.sort(key=lambda item: (-item[1], item[0]))

        # 4. Agregación Documental (Max Pooling sobre RRF) para F1@3
        doc_scores: Dict[str, float] = {}
        for idx, score in fused_scores:
            doc_id = self.metadatos[idx]["doc_id"]
            if doc_id not in doc_scores or score > doc_scores[doc_id]:
                doc_scores[doc_id] = score

        # Completa documentos con IDs únicos y score cero cuando ranking no
        # alcanza cardinalidad. Nunca duplica documentos del índice.
        doc_order: List[str] = []
        for meta in self.metadatos:
            doc_id = meta.get("doc_id", "")
            if doc_id and doc_id not in doc_order:
                doc_order.append(doc_id)

        top_docs_sorted = sorted(
            doc_scores.items(), key=lambda x: (-x[1], x[0])
        )[:top_k_docs]
        docs_seleccionados = {doc_id for doc_id, _ in top_docs_sorted}
        docs_rellenados: List[str] = []
        for doc_id in doc_order:
            if len(top_docs_sorted) >= top_k_docs:
                break
            if doc_id not in docs_seleccionados:
                top_docs_sorted.append((doc_id, 0.0))
                docs_seleccionados.add(doc_id)
                docs_rellenados.append(doc_id)

        documents_output = [
            {"rank": rank + 1, "doc_id": doc_id}
            for rank, (doc_id, _) in enumerate(top_docs_sorted)
        ]

        # 5. Deduplicación por Jaccard y Recorte ≤ 250 palabras para NDCG@10
        chunks_seleccionados: List[Dict[str, Any]] = []
        textos_vistos_por_doc: Dict[str, List[str]] = {}

        indices_ordenados = list(fused_scores)
        indices_ordenados.extend(
            (idx, 0.0)
            for idx, meta in enumerate(self.metadatos)
            if meta.get("doc_id") in docs_seleccionados
            and idx not in {candidate_idx for candidate_idx, _ in fused_scores}
        )
        chunks_rellenados = 0

        for idx, score in indices_ordenados:
            if len(chunks_seleccionados) >= top_k_chunks:
                break

            meta = self.metadatos[idx]
            doc_id = meta["doc_id"]
            if doc_id not in docs_seleccionados:
                continue
            chunk_id = meta.get("chunk_id", "")
            if not chunk_id:
                continue
            if any(f["chunk_id"] == chunk_id for f in chunks_seleccionados):
                continue
            texto = meta["texto"]

            # Evitar solape excesivo con chunks ya seleccionados del mismo documento
            vistos = textos_vistos_por_doc.setdefault(doc_id, [])
            es_duplicado = any(self._jaccard_similarity(texto, v) > 0.5 for v in vistos)
            if es_duplicado:
                continue

            vistos.append(texto)
            texto_recortado = self._recortar_a_250_palabras(texto, max_words=250)
            if score == 0.0:
                chunks_rellenados += 1

            chunks_seleccionados.append({
                "rank": len(chunks_seleccionados) + 1,
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text": texto_recortado,
                "score_fusion": float(score),
                "score_bm25_raw": float(bm25_scores.get(idx, 0.0)),
                "score_faiss_raw": float(faiss_scores.get(idx, 0.0)),
                "score_bm25_normalized": float(bm25_normalizados.get(idx, 0.0)),
                "score_faiss_normalized": float(faiss_normalizados.get(idx, 0.0)),
                "rank_bm25": bm25_ranks.get(idx),
                "rank_faiss": faiss_ranks.get(idx),
                "fuente": meta.get("fuente", ""),
                "num_tokens": meta.get("num_tokens", 0),
                "padding": score == 0.0,
            })

        return {
            "documents": documents_output,
            "fragments": chunks_seleccionados,
            "diagnostics": {
                "document_candidates": len(doc_scores),
                "padded_documents": docs_rellenados,
                "padded_fragments": chunks_rellenados,
                "documents_requested": top_k_docs,
                "fragments_requested": top_k_chunks,
                "fusion_method": fusion_method,
                "normalization_method": normalization_method,
                "alpha": alpha,
            },
        }


def contar_palabras(texto: str) -> int:
    """Cuenta palabras según contrato de salida y validación."""
    return len(texto.split())
