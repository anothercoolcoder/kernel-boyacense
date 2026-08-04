"""Indexación vectorial: etapa 3 del pipeline RAG.

Recibe la lista de :class:`Fragmento` producida por :mod:`fragmentacion` y la
convierte en un índice FAISS listo para búsqueda semántica.

Cambios respecto al borrador original:
- Sin LangChain splitters: los fragmentos ya llegan segmentados y con metadata.
- Sin lectura de `.md` intermedios: los objetos viajan en RAM.
- Detección automática de acelerador: CUDA > CPU.
- Sin ``CARPETA_ENTRADA`` fija; la ruta de corpus la fija el orquestador.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

FAISS_INDEX_DIR = "./faiss_index"
MODELO_EMBEDDINGS = "intfloat/multilingual-e5-large-instruct"


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


def fragmentos_a_documentos(fragmentos: List[dict]) -> List[Document]:
    """Convierte la lista de Fragmento del módulo fragmentacion en Documents LangChain.

    No aplica ningún corte adicional: los fragmentos ya tienen el tamaño
    correcto para el tokenizador del modelo de embeddings.

    Lee el esquema estándar de chunks (alineado con chunks.jsonl del pipeline):
    ``doc_id``, ``chunk_id``, ``fuente``, ``formato``, ``fenomeno``,
    ``posicion``, ``num_tokens``.

    Args:
        fragmentos: lista de dicts producida por :func:`fragmentacion.fragmentar_registros`.

    Returns:
        Lista de :class:`langchain_core.documents.Document` con metadata nativa.
    """
    documentos: List[Document] = []
    for idx, frag in enumerate(fragmentos):
        # Mapeo flexible para metadatos anidados en _meta o metadata
        meta_aux = frag.get("_meta") or frag.get("metadata") or {}

        # 1. Obtener fuente desde la raíz o desde la metainformación anidada
        fuente = (
            frag.get("fuente")
            or meta_aux.get("fuente")
            or frag.get("source")
            or meta_aux.get("source")
            or "Desconocido"
        )
        if isinstance(fuente, Path):
            fuente = str(fuente)

        # 2. Obtener o autogenerar doc_id a partir del nombre del archivo
        doc_id = frag.get("doc_id") or meta_aux.get("doc_id")
        if not doc_id:
            nombre_base = os.path.basename(fuente) if fuente != "Desconocido" else "doc"
            doc_id = hashlib.md5(nombre_base.encode("utf-8")).hexdigest()[:8]

        # 3. Obtener o autogenerar chunk_id secuencial
        chunk_id = frag.get("chunk_id") or meta_aux.get("chunk_id") or f"{doc_id}_chunk_{idx}"

        metadata = {
            "doc_id":    doc_id,
            "chunk_id":  chunk_id,
            "fuente":    fuente,
            "formato":   frag.get("formato") or meta_aux.get("formato", ""),
            "fenomeno":  frag.get("fenomeno") if frag.get("fenomeno") is not None else meta_aux.get("fenomeno"),
            "posicion":  frag.get("posicion") or meta_aux.get("posicion", idx),
            "num_tokens": frag.get("num_tokens") or meta_aux.get("num_tokens", 0),
            # Trazabilidad auxiliar (pagina, idioma, etc.)
            **meta_aux,
        }
        
        texto = frag.get("texto") or frag.get("page_content", "")
        documentos.append(Document(page_content=texto, metadata=metadata))
        
    return documentos


def indexar(fragmentos: List[dict], faiss_dir: str = FAISS_INDEX_DIR) -> None:
    """Crea y persiste el índice FAISS a partir de la lista de Fragmento.

    Args:
        fragmentos: lista de dicts producida por :func:`fragmentacion.fragmentar_registros`.
        faiss_dir:  directorio donde se guarda el índice; se crea si no existe.
    """
    if not fragmentos:
        logger.warning("Sin fragmentos para indexar. El índice no se creará.")
        return

    documentos = fragmentos_a_documentos(fragmentos)
    logger.info("Indexando %d documentos LangChain…", len(documentos))

    device = _detectar_device()
    embeddings = HuggingFaceEmbeddings(
        model_name=MODELO_EMBEDDINGS,
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32,
        },
    )

    vector_store = FAISS.from_documents(documents=documentos, embedding=embeddings)
    vector_store.save_local(faiss_dir)
    logger.info("Índice FAISS guardado en: %s", faiss_dir)
    print(f"Indexación FAISS completada. Guardado en: '{faiss_dir}'")