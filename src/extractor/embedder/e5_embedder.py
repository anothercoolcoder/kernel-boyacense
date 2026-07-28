"""
Generación de embeddings con multilingual-e5-base.

Modelo: intfloat/multilingual-e5-base
- Arquitectura: XLM-RoBERTa (tipo BERT)
- Dimensionalidad: 768
- Max tokens: 512
- Idiomas: 100+ incluyendo ES, PT, EN
- Requiere prefijo "passage: " para documentos y "query: " para consultas
"""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "intfloat/multilingual-e5-base"
_EMBEDDING_DIM = 768


class E5Embedder:
    """
    Genera embeddings con multilingual-e5-base.

    Parameters
    ----------
    model_name : str
        Nombre del modelo de HuggingFace a usar.
    batch_size : int
        Tamaño de batch para codificación. Default: 32.
    device : str or None
        Dispositivo ("cpu", "cuda", "mps"). None = auto-detect.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size

        logger.info("🧠 Cargando modelo de embeddings '%s'…", model_name)
        self._model = SentenceTransformer(model_name, device=device)

        actual_dim = self._model.get_embedding_dimension()
        logger.info(
            "  └─ Modelo cargado — dim: %d, device: %s",
            actual_dim,
            self._model.device,
        )

        if actual_dim != _EMBEDDING_DIM:
            logger.warning(
                "  ⚠️  Dimensionalidad esperada: %d, obtenida: %d",
                _EMBEDDING_DIM,
                actual_dim,
            )

    @property
    def dimension(self) -> int:
        """Dimensionalidad de los embeddings generados."""
        return self._model.get_embedding_dimension()

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """
        Genera embeddings para textos de documentos (passages).

        Añade el prefijo "passage: " requerido por E5.

        Parameters
        ----------
        texts : list[str]
            Lista de textos a codificar.

        Returns
        -------
        np.ndarray
            Matriz de embeddings con shape (n, 768).
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # E5 requiere prefijo para passages
        prefixed = [f"passage: {t}" for t in texts]

        logger.info(
            "  ├─ Generando embeddings para %d textos (batch_size=%d)…",
            len(texts),
            self.batch_size,
        )

        embeddings = self._model.encode(
            prefixed,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # normalizar para cosine similarity
            convert_to_numpy=True,
        )

        logger.info(
            "  └─ Embeddings generados: shape %s, dtype %s",
            embeddings.shape,
            embeddings.dtype,
        )

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Genera embedding para una consulta de búsqueda.

        Añade el prefijo "query: " requerido por E5.

        Parameters
        ----------
        query : str
            Texto de la consulta.

        Returns
        -------
        np.ndarray
            Vector de embedding con shape (768,).
        """
        prefixed = f"query: {query}"
        embedding = self._model.encode(
            [prefixed],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding[0]
