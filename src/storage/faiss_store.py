"""
Almacenamiento provisional en FAISS.

Implementación sencilla para la Fase 1 del proyecto.
Usa IndexFlatIP (Inner Product) con vectores normalizados = cosine similarity.
"""

import json
import logging
from pathlib import Path

import faiss
import numpy as np

from extractor.models import ChunkResult, SearchResult

logger = logging.getLogger(__name__)

_METADATA_FILENAME = "metadata.json"
_INDEX_FILENAME = "index.faiss"


class FAISSStore:
    """
    Almacén vectorial basado en FAISS (provisional).

    Utiliza IndexFlatIP para búsqueda exacta por inner product.
    Con vectores normalizados L2, equivale a cosine similarity.

    Parameters
    ----------
    dimension : int
        Dimensionalidad de los embeddings. Default: 768.
    """

    def __init__(self, dimension: int = 768) -> None:
        self.dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._metadata: dict[int, dict] = {}  # faiss_id → metadata + text
        self._next_id: int = 0

        logger.info(
            "💾 FAISSStore inicializado — dim: %d, tipo: IndexFlatIP",
            dimension,
        )

    @property
    def size(self) -> int:
        """Cantidad de vectores almacenados."""
        return self._index.ntotal

    def add(self, chunks: list[ChunkResult]) -> None:
        """
        Agrega chunks al índice FAISS.

        Parameters
        ----------
        chunks : list[ChunkResult]
            Chunks con embeddings a almacenar.
        """
        if not chunks:
            logger.warning("⚠️  Lista de chunks vacía, nada que agregar.")
            return

        # Construir matriz de embeddings
        embeddings = np.stack(
            [c.embedding for c in chunks]
        ).astype(np.float32)

        # Normalizar (por seguridad, aunque el embedder ya normaliza)
        faiss.normalize_L2(embeddings)

        # Agregar al índice
        start_id = self._next_id
        self._index.add(embeddings)

        # Guardar metadata
        for i, chunk in enumerate(chunks):
            faiss_id = start_id + i
            self._metadata[faiss_id] = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                **chunk.metadata,
            }

        self._next_id = start_id + len(chunks)

        logger.info(
            "  ├─ %d vectores agregados al índice (total: %d)",
            len(chunks),
            self.size,
        )

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> list[SearchResult]:
        """
        Busca los k vecinos más cercanos a un query embedding.

        Parameters
        ----------
        query_embedding : np.ndarray
            Vector de consulta (768-d).
        k : int
            Número de resultados. Default: 5.

        Returns
        -------
        list[SearchResult]
            Resultados ordenados por relevancia descendente.
        """
        if self.size == 0:
            logger.warning("⚠️  Índice vacío, no se puede buscar.")
            return []

        # Asegurar forma correcta
        query = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)

        k = min(k, self.size)
        distances, indices = self._index.search(query, k)

        results: list[SearchResult] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            meta = self._metadata.get(int(idx), {})
            results.append(SearchResult(
                chunk_id=meta.get("chunk_id", "unknown"),
                text=meta.get("text", ""),
                score=float(score),
                metadata={
                    k: v for k, v in meta.items()
                    if k not in ("chunk_id", "text")
                },
            ))

        logger.info(
            "🔍 Búsqueda completada — %d resultados (mejor score: %.4f)",
            len(results),
            results[0].score if results else 0.0,
        )

        return results

    def save(self, output_dir: str | Path) -> None:
        """
        Persiste el índice y metadata a disco.

        Parameters
        ----------
        output_dir : str or Path
            Directorio donde guardar los archivos.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        index_path = output_dir / _INDEX_FILENAME
        metadata_path = output_dir / _METADATA_FILENAME

        # Guardar índice FAISS
        faiss.write_index(self._index, str(index_path))
        logger.info("  ├─ Índice FAISS guardado: %s", index_path)

        # Guardar metadata como JSON
        # Convertir claves int a string para JSON
        serializable = {str(k): v for k, v in self._metadata.items()}
        metadata_path.write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("  ├─ Metadata guardada: %s", metadata_path)
        logger.info(
            "  └─ Total guardado: %d vectores, %d entradas de metadata",
            self.size,
            len(self._metadata),
        )

    def load(self, input_dir: str | Path) -> None:
        """
        Carga índice y metadata desde disco.

        Parameters
        ----------
        input_dir : str or Path
            Directorio desde donde cargar.
        """
        input_dir = Path(input_dir)

        index_path = input_dir / _INDEX_FILENAME
        metadata_path = input_dir / _METADATA_FILENAME

        if not index_path.exists():
            raise FileNotFoundError(f"No se encontró índice: {index_path}")

        # Cargar índice
        self._index = faiss.read_index(str(index_path))
        logger.info("  ├─ Índice FAISS cargado: %d vectores", self._index.ntotal)

        # Cargar metadata
        if metadata_path.exists():
            raw = json.loads(metadata_path.read_text(encoding="utf-8"))
            self._metadata = {int(k): v for k, v in raw.items()}
            self._next_id = max(self._metadata.keys(), default=-1) + 1
            logger.info(
                "  └─ Metadata cargada: %d entradas", len(self._metadata)
            )
        else:
            logger.warning("  └─ ⚠️  No se encontró archivo de metadata.")
            self._metadata = {}
            self._next_id = self._index.ntotal
