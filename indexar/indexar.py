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
import os
import tempfile
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
    """Devuelve 'cuda' si hay GPU NVIDIA disponible, 'cpu' en caso contrario."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("GPU NVIDIA detectada — usando CUDA para embeddings.")
            return "cuda"
    except ImportError:
        pass

    logger.info("Sin GPU CUDA disponible — usando CPU para embeddings.")
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
        texto_raw = frag["texto"]
        num_tokens = frag.get("num_tokens", 0)

        # 1. Filtro de tokens mínimos
        if num_tokens < 20:
            continue

        # 2. Filtro de texto basura (ruido gráfico OCR o tablas de código/números pegados)
        tokens_texto = texto_raw.split()
        if tokens_texto:
            palabras_muy_largas = sum(1 for t in tokens_texto if len(t) > 25)
            if palabras_muy_largas / len(tokens_texto) > 0.35:
                continue

        textos.append(f"passage: {texto_raw}")

        meta = {
            "doc_id":     frag.get("doc_id", ""),
            "chunk_id":   frag.get("chunk_id", ""),
            "fuente":     frag.get("fuente", ""),
            "formato":    frag.get("formato", ""),
            "fenomeno":   frag.get("fenomeno"),
            "posicion":   frag.get("posicion", 0),
            "num_tokens": num_tokens,
            "texto":      texto_raw,
        }
        meta.update(
            {
                clave: valor
                for clave, valor in frag.get("_meta", {}).items()
                if clave not in meta
            }
        )
        metadatos.append(meta)

    # El filtro de chunks cortos puede abrir huecos en posiciones generadas
    # antes de indexar. Metadata final debe conservar ordinal 0-based continuo.
    posiciones_por_doc: dict[str, int] = {}
    for meta in metadatos:
        doc_id = meta["doc_id"]
        meta["posicion"] = posiciones_por_doc.get(doc_id, 0)
        posiciones_por_doc[doc_id] = meta["posicion"] + 1

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
    if device == "cpu":
        try:
            import torch
            cpus = os.cpu_count() or 4
            torch.set_num_threads(cpus)
            logger.info("Optimizado PyTorch para usar %d hilos de CPU", cpus)
        except Exception:
            pass

    embeddings_model = HuggingFaceEmbeddings(
        model_name=MODELO_EMBEDDINGS,
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32,
        },
    )

    # Generar vectores de embeddings con barra de progreso explícita tqdm lote por lote
    from tqdm import tqdm
    batch_size = 32
    vectors_list = []
    for i in tqdm(range(0, len(textos), batch_size), desc="Etapa 3: Generando embeddings FAISS", unit="lote"):
        lote = textos[i : i + batch_size]
        vecs = embeddings_model._client.encode(
            lote,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        vectors_list.append(vecs)

    vectors_np = np.vstack(vectors_list)

    # Crear índice FAISS (Inner Product ≡ coseno con vectores normalizados)
    dimension = vectors_np.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors_np)

    # chunk_id pertenece al contrato documental y ya es determinista desde
    # fragmentación. El ordinal FAISS es interno y no debe sobrescribirlo.
    faiss_path = out_dir / "index.faiss"
    metadata_path = out_dir / "metadata.jsonl"
    with tempfile.TemporaryDirectory(prefix=".index-build-", dir=out_dir) as temporal:
        faiss_tmp = Path(temporal) / "index.faiss"
        metadata_tmp = Path(temporal) / "metadata.jsonl"
        faiss.write_index(index, str(faiss_tmp))
        _escribir_metadata_jsonl(metadatos, metadata_tmp)
        for temporal_path in (faiss_tmp, metadata_tmp):
            with temporal_path.open("rb") as archivo:
                os.fsync(archivo.fileno())
        os.replace(faiss_tmp, faiss_path)
        os.replace(metadata_tmp, metadata_path)
    logger.info("Índice FAISS guardado atómicamente: %s (%d vectores, dim=%d)", faiss_path, index.ntotal, dimension)

    print(f"Indexación FAISS completada. Guardado en: '{faiss_dir}'")
    print(f"  -> index.faiss: {index.ntotal} vectores (dim={dimension})")
    print(f"  -> metadata.jsonl: {len(metadatos)} registros")
