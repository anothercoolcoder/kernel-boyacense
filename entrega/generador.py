"""Generador reproducible del entregable CODEFEST AD ASTRA 2026.

El archivo es autocontenido: carga FAISS y metadata, ejecuta BM25 + FAISS,
fusiona con CombSUM min-max y genera resultados.jsonl. No usa LLM generativo.
Configuracion seleccionada en benchmark: alpha=0.1.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_VECTORIAL = ROOT / "base_vectorial"
DEFAULT_QUERIES = ROOT / "consultas.jsonl"
DEFAULT_OUTPUT = ROOT / "resultados.jsonl"
MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
QUERY_INSTRUCTION = (
    "Given a question, retrieve passages from documents that contain the exact "
    "factual information needed to answer the question"
)
TOP_K_DOCS = 3
TOP_K_CHUNKS = 10
CANDIDATE_K = 60
FUSION_METHOD = "combsum"
NORMALIZATION_METHOD = "minmax"
ALPHA = 0.1

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generador")


def contar_palabras(texto: str) -> int:
    return len(texto.split())


class BM25Nativo:
    """BM25Okapi minimo, sin dependencia de un servicio externo."""

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_tokens = [self._tokenize(texto) for texto in corpus]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / max(1, self.corpus_size)
        self.doc_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.idf: dict[str, float] = {}
        self._calculate_idf()

    @staticmethod
    def _tokenize(texto: str) -> list[str]:
        texto = re.sub(r"[^\w\s]", " ", texto.lower())
        return [token for token in texto.split() if len(token) > 1]

    def _calculate_idf(self) -> None:
        frecuencias: Counter[str] = Counter()
        for documento in self.doc_freqs:
            frecuencias.update(documento.keys())
        for token, frecuencia in frecuencias.items():
            self.idf[token] = math.log(
                (self.corpus_size - frecuencia + 0.5) / (frecuencia + 0.5) + 1.0
            )

    def search(self, query: str, top_k: int = CANDIDATE_K) -> list[tuple[int, float]]:
        tokens_query = self._tokenize(query)
        if not tokens_query:
            return []
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        for token in tokens_query:
            if token not in self.idf:
                continue
            for indice, frecuencias in enumerate(self.doc_freqs):
                frecuencia = frecuencias.get(token, 0)
                if frecuencia == 0:
                    continue
                numerador = frecuencia * (self.k1 + 1)
                denominador = frecuencia + self.k1 * (
                    1 - self.b + self.b * (self.doc_len[indice] / self.avgdl)
                )
                scores[indice] += self.idf[token] * numerador / denominador
        indices = np.argsort(scores)[::-1][:top_k]
        return [(int(indice), float(scores[indice])) for indice in indices if scores[indice] > 0]


def resolver_encoder(base_vectorial: Path) -> Path:
    """Encuentra encoder único desde raíz base_vectorial."""
    candidatos = sorted(
        ruta
        for ruta in base_vectorial.glob("encoder_*")
        if ruta.is_dir()
        and (ruta / "index.faiss").is_file()
        and (ruta / "metadata.jsonl").is_file()
    )
    if not candidatos:
        raise FileNotFoundError(
            f"No se encontró encoder con index.faiss y metadata.jsonl en {base_vectorial}"
        )
    preferido = base_vectorial / "encoder_multilingual-e5-large-instruct"
    if preferido in candidatos:
        return preferido
    if len(candidatos) > 1:
        nombres = ", ".join(ruta.name for ruta in candidatos)
        raise ValueError(f"Base vectorial ambigua; encoders encontrados: {nombres}")
    return candidatos[0]


class Recuperador:
    """Estrategia completa de recuperacion usada por benchmark ganador."""

    def __init__(self, encoder_dir: Path):
        self.encoder_dir = Path(encoder_dir)
        index_path = self.encoder_dir / "index.faiss"
        metadata_path = self.encoder_dir / "metadata.jsonl"
        if not index_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"Faltan index.faiss o metadata.jsonl en {self.encoder_dir}"
            )

        import faiss

        self.faiss_index = faiss.read_index(str(index_path))
        self.metadatos: list[dict[str, Any]] = []
        with metadata_path.open(encoding="utf-8") as archivo:
            for numero, linea in enumerate(archivo, 1):
                if linea.strip():
                    registro = json.loads(linea)
                    if not isinstance(registro, dict):
                        raise ValueError(f"Metadata invalida en linea {numero}")
                    self.metadatos.append(registro)
        if self.faiss_index.ntotal != len(self.metadatos):
            raise ValueError("Cantidad de vectores FAISS no coincide con metadata.jsonl")
        self.bm25 = BM25Nativo([meta.get("texto", "") for meta in self.metadatos])
        self.embeddings_model = None

    @staticmethod
    def normalizar_scores(scores: dict[int, float]) -> dict[int, float]:
        if not scores:
            return {}
        minimo = min(scores.values())
        maximo = max(scores.values())
        if minimo == maximo:
            return {indice: 1.0 for indice in scores}
        return {
            indice: float(np.clip((score - minimo) / (maximo - minimo), 0.0, 1.0))
            for indice, score in scores.items()
        }

    @staticmethod
    def recortar_a_250_palabras(texto: str, max_words: int = 250) -> str:
        if contar_palabras(texto) <= max_words:
            return texto
        oraciones = re.split(r"(?<=[.!?])\s+", texto.strip())
        resultado: list[str] = []
        palabras = 0
        for oracion in oraciones:
            cantidad = contar_palabras(oracion)
            if not oracion or palabras + cantidad > max_words:
                break
            resultado.append(oracion)
            palabras += cantidad
        return " ".join(resultado) if resultado else " ".join(texto.split()[:max_words])

    @staticmethod
    def jaccard(texto_a: str, texto_b: str) -> float:
        tokens_a = set(texto_a.lower().split())
        tokens_b = set(texto_b.lower().split())
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    def _cargar_modelo(self) -> None:
        if self.embeddings_model is None:
            from langchain_huggingface import HuggingFaceEmbeddings

            self.embeddings_model = HuggingFaceEmbeddings(
                model_name=MODEL_NAME,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

    def buscar(
        self,
        pregunta: str,
        top_k_docs: int = TOP_K_DOCS,
        top_k_chunks: int = TOP_K_CHUNKS,
    ) -> dict[str, Any]:
        self._cargar_modelo()
        bm25 = self.bm25.search(pregunta, CANDIDATE_K)
        bm25_ranks = {indice: rango + 1 for rango, (indice, _) in enumerate(bm25)}
        bm25_scores = {indice: score for indice, score in bm25}

        query = f"Instruct: {QUERY_INSTRUCTION}\nQuery: {pregunta}"
        vector = np.asarray([self.embeddings_model.embed_query(query)], dtype=np.float32)
        scores_faiss, indices_faiss = self.faiss_index.search(vector, CANDIDATE_K)
        faiss_ranks = {
            int(indice): rango + 1
            for rango, indice in enumerate(indices_faiss[0])
            if indice >= 0
        }
        faiss_scores = {
            int(indice): float(score)
            for score, indice in zip(scores_faiss[0], indices_faiss[0])
            if indice >= 0
        }

        bm25_norm = self.normalizar_scores(bm25_scores)
        faiss_norm = self.normalizar_scores(faiss_scores)
        candidatos = set(bm25_ranks) | set(faiss_ranks)
        fusionados = sorted(
            (
                indice,
                ALPHA * bm25_norm.get(indice, 0.0)
                + (1.0 - ALPHA) * faiss_norm.get(indice, 0.0),
            )
            for indice in candidatos
        )
        fusionados.sort(key=lambda item: (-item[1], item[0]))

        scores_documento: dict[str, float] = {}
        for indice, score in fusionados:
            doc_id = self.metadatos[indice]["doc_id"]
            scores_documento[doc_id] = max(score, scores_documento.get(doc_id, 0.0))
        documentos = sorted(scores_documento.items(), key=lambda item: (-item[1], item[0]))
        documentos = documentos[:top_k_docs]
        seleccionados = {doc_id for doc_id, _ in documentos}

        # Completa cardinalidad solo con documentos existentes en metadata.
        for meta in self.metadatos:
            doc_id = meta.get("doc_id", "")
            if len(documentos) >= top_k_docs:
                break
            if doc_id and doc_id not in seleccionados:
                documentos.append((doc_id, 0.0))
                seleccionados.add(doc_id)

        documentos_salida = [
            {"rank": rango + 1, "doc_id": doc_id}
            for rango, (doc_id, _) in enumerate(documentos)
        ]
        chunks: list[dict[str, Any]] = []
        vistos_por_doc: dict[str, list[str]] = {}
        indices_fusionados = {indice for indice, _ in fusionados}
        candidatos_chunks = list(fusionados)
        candidatos_chunks.extend(
            (indice, 0.0)
            for indice, meta in enumerate(self.metadatos)
            if meta.get("doc_id") in seleccionados and indice not in indices_fusionados
        )
        for indice, score in candidatos_chunks:
            if len(chunks) >= top_k_chunks:
                break
            meta = self.metadatos[indice]
            doc_id = meta.get("doc_id", "")
            chunk_id = meta.get("chunk_id", "")
            texto = meta.get("texto", "")
            if doc_id not in seleccionados or not chunk_id:
                continue
            if any(item["chunk_id"] == chunk_id for item in chunks):
                continue
            vistos = vistos_por_doc.setdefault(doc_id, [])
            if any(self.jaccard(texto, anterior) > 0.5 for anterior in vistos):
                continue
            vistos.append(texto)
            chunks.append(
                {
                    "rank": len(chunks) + 1,
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "text": self.recortar_a_250_palabras(texto),
                }
            )
        return {"documents": documentos_salida, "fragments": chunks}


def normalizar_query_id(valor: Any) -> str:
    """Normaliza IDs numéricos o qN al contrato qNNN."""
    texto = str(valor).strip()
    coincidencia = re.fullmatch(r"q?(\d+)", texto, flags=re.IGNORECASE)
    if coincidencia:
        return f"q{int(coincidencia.group(1)):03d}"
    return texto


def cargar_consultas(path: Path) -> list[dict[str, str]]:
    """Carga consultas JSONL tolerando nombres razonables de campos."""
    consultas: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as archivo:
        for numero, linea in enumerate(archivo, 1):
            if not linea.strip():
                continue
            try:
                item = json.loads(linea)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON invalido en linea {numero}: {exc.msg}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Consulta invalida en linea {numero}: se esperaba objeto JSON")

            valor_id = item.get("query_id", item.get("id"))
            if valor_id is None or not str(valor_id).strip():
                campos = ", ".join(sorted(item)) or "ninguno"
                raise ValueError(
                    f"Linea {numero}: falta identificador; campos encontrados: {campos}"
                )
            query_id = normalizar_query_id(valor_id)

            texto = next(
                (item[campo] for campo in ("query", "pregunta", "consulta", "text")
                 if campo in item and isinstance(item[campo], str) and item[campo].strip()),
                None,
            )
            if texto is None:
                campos = ", ".join(sorted(item)) or "ninguno"
                raise ValueError(
                    f"Linea {numero}: falta texto de consulta; campos encontrados: {campos}; "
                    "campos aceptados: query, pregunta, consulta, text"
                )
            consultas.append({"query_id": query_id, "query": texto.strip()})
    return consultas


def validar_resultados(resultados: list[dict[str, Any]]) -> None:
    if len(resultados) != 50:
        raise ValueError(f"Se requieren exactamente 50 consultas; hay {len(resultados)}")
    ids_esperados = [f"q{indice:03d}" for indice in range(1, 51)]
    ids_actuales = [resultado.get("query_id") for resultado in resultados]
    if ids_actuales != ids_esperados:
        raise ValueError("Consultas deben estar ordenadas q001...q050")
    for resultado in resultados:
        query_id = resultado["query_id"]
        documentos = resultado.get("documents", [])
        fragmentos = resultado.get("fragments", [])
        if len(documentos) != TOP_K_DOCS or len(fragmentos) != TOP_K_CHUNKS:
            raise ValueError(f"{query_id}: cardinalidad invalida")
        doc_ids = [doc["doc_id"] for doc in documentos]
        chunk_ids = [frag["chunk_id"] for frag in fragmentos]
        if len(set(doc_ids)) != 3 or len(set(chunk_ids)) != 10:
            raise ValueError(f"{query_id}: IDs duplicados")
        if any(frag["doc_id"] not in doc_ids for frag in fragmentos):
            raise ValueError(f"{query_id}: fragmento fuera de documents")
        if any(contar_palabras(frag["text"]) > 250 for frag in fragmentos):
            raise ValueError(f"{query_id}: fragmento excede 250 palabras")


def generar(queries_path: Path, output_path: Path, base_vectorial: Path) -> None:
    if not queries_path.is_file():
        raise FileNotFoundError(
            f"No existe archivo de consultas: {queries_path}. "
            "Use --consultas con dataset oficial q001...q050."
        )
    consultas = cargar_consultas(queries_path)
    ids_esperados = [f"q{n:03d}" for n in range(1, 51)]
    if [item["query_id"] for item in consultas] != ids_esperados:
        raise ValueError("Archivo de consultas debe contener q001...q050 en orden")
    encoder_dir = resolver_encoder(base_vectorial)
    logger.info("Encoder seleccionado: %s", encoder_dir.name)
    recuperador = Recuperador(encoder_dir)
    resultados = []
    for consulta in consultas:
        logger.info("Procesando %s", consulta["query_id"])
        recuperado = recuperador.buscar(consulta["query"])
        resultados.append({"query_id": consulta["query_id"], **recuperado})
    validar_resultados(resultados)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as archivo:
        for resultado in resultados:
            archivo.write(json.dumps(resultado, ensure_ascii=False) + "\n")
    logger.info("Generado %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consultas", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--salida", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-vectorial", type=Path, default=DEFAULT_BASE_VECTORIAL)
    args = parser.parse_args()
    generar(args.consultas, args.salida, args.base_vectorial)


if __name__ == "__main__":
    main()
