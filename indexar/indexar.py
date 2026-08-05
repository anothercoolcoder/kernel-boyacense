"""Indexación vectorial: etapa 3 del pipeline RAG.

Recibe la lista de :class:`Fragmento` producida por :mod:`fragmentacion` y la
convierte en un índice FAISS listo para búsqueda semántica.

Cambios respecto al borrador original:
- Sin LangChain splitters: los fragmentos ya llegan segmentados y con metadata.
- Sin lectura de `.md` intermedios: los objetos viajan en RAM.
- Detección automática de acelerador: CUDA > CPU.
- Sin ``CARPETA_ENTRADA`` fija; la ruta de corpus la fija el orquestador.

Formato de salida alineado con el CODEFEST AD ASTRA 2026:
- ``index.faiss``: índice FAISS puro (serializado con ``faiss.write_index``).
- ``metadata.jsonl``: almacén de metadata en JSON Lines, 1 objeto por línea.
  El orden de líneas coincide con los IDs internos asignados por FAISS.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

ENCODER_NAME = "multilingual-e5-large-instruct"
MODELO_EMBEDDINGS = f"intfloat/{ENCODER_NAME}"
BASE_VECTORIAL_DIR = f"./base_vectorial/encoder_{ENCODER_NAME}"


def _detectar_device() -> str:
    """Devuelve 'cuda' si hay GPU disponible, 'cpu' en caso contrario."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("GPU detectada — usando CUDA para embeddings.")
            return "cuda"
    except ImportError:
        pass
    logger.info("Sin GPU disponible — usando CPU para embeddings.")
    return "cpu"


def _preparar_textos(fragmentos: List[dict]) -> tuple[List[str], List[dict]]:
    """Extrae textos y registros de metadata alineados de la lista de fragmentos.

    Lee el esquema estándar de chunks (alineado con chunks.jsonl del pipeline):
    ``doc_id``, ``chunk_id``, ``fuente``, ``formato``, ``fenomeno``,
    ``posicion``, ``num_tokens``.

    Args:
        fragmentos: lista de dicts producida por :func:`fragmentacion.fragmentar_registros`.

    Returns:
        Tupla (textos, metadatos) donde:
        - textos: lista de strings con el contenido de cada fragmento.
        - metadatos: lista de dicts con los campos obligatorios de metadata,
          en el mismo orden que los textos (y que los IDs internos de FAISS).
    """
    textos: List[str] = []
    metadatos: List[dict] = []

    for frag in fragmentos:
        textos.append(frag["texto"])

        meta = {
            "doc_id":     frag.get("doc_id", ""),
            "chunk_id":   frag.get("chunk_id", ""),
            "fuente":     frag.get("fuente", ""),
            "formato":    frag.get("formato", ""),
            "fenomeno":   frag.get("fenomeno"),
            "posicion":   frag.get("posicion", 0),
            "num_tokens": frag.get("num_tokens", 0),
            "texto":      frag["texto"],
        }
        metadatos.append(meta)

    return textos, metadatos


def _escribir_metadata_jsonl(metadatos: List[dict], ruta: Path) -> None:
    """Persiste la lista de metadata como JSON Lines.

    El orden de las líneas coincide exactamente con los IDs internos de FAISS
    (línea 0 = vector con FAISS ID 0, etc.).
    """
    with open(ruta, "w", encoding="utf-8") as f:
        for meta in metadatos:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    logger.info("Metadata JSONL escrito: %s (%d registros)", ruta, len(metadatos))


def indexar(fragmentos: List[dict], faiss_dir: str = BASE_VECTORIAL_DIR) -> None:
    """Crea y persiste el índice FAISS + metadata.jsonl a partir de la lista de Fragmento.

    Genera:
    - ``<faiss_dir>/index.faiss``: índice FAISS puro (IndexFlatIP, coseno con
      vectores normalizados), serializado con ``faiss.write_index()``.
    - ``<faiss_dir>/metadata.jsonl``: almacén de metadata en JSON Lines.

    Args:
        fragmentos: lista de dicts producida por :func:`fragmentacion.fragmentar_registros`.
        faiss_dir:  directorio donde se guarda el índice; se crea si no existe.
    """
    if not fragmentos:
        logger.warning("Sin fragmentos para indexar. El índice no se creará.")
        return

    out_dir = Path(faiss_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    textos, metadatos = _preparar_textos(fragmentos)
    logger.info("Generando embeddings para %d fragmentos…", len(textos))

    device = _detectar_device()
    embeddings_model = HuggingFaceEmbeddings(
        model_name=MODELO_EMBEDDINGS,
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32,
        },
    )

    # Generar vectores de embeddings
    vectors = embeddings_model.embed_documents(textos)
    vectors_np = np.array(vectors, dtype=np.float32)

    # Crear índice FAISS (Inner Product ≡ coseno con vectores normalizados)
    dimension = vectors_np.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors_np)

    # Persistir índice FAISS puro
    faiss_path = out_dir / "index.faiss"
    faiss.write_index(index, str(faiss_path))
    logger.info("Índice FAISS guardado: %s (%d vectores, dim=%d)", faiss_path, index.ntotal, dimension)

    # Persistir metadata JSONL
    _escribir_metadata_jsonl(metadatos, out_dir / "metadata.jsonl")

    print(f"Indexación FAISS completada. Guardado en: '{faiss_dir}'")
    print(f"  → index.faiss: {index.ntotal} vectores (dim={dimension})")
    print(f"  → metadata.jsonl: {len(metadatos)} registros")