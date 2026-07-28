"""
Modelos de datos compartidos por todos los sub-módulos del extractor.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import uuid

import numpy as np


@dataclass
class Document:
    """Resultado de leer un archivo — texto crudo + metadatos de origen."""

    text: str
    metadata: dict = field(default_factory=dict)

    # ── Campos de metadata esperados ────────────────────────────────
    # source_file : str   — ruta absoluta del archivo fuente
    # file_type   : str   — extensión (.pdf, .html, …)
    # page        : int   — número de página (si aplica)
    # title       : str   — título del documento (si lo tiene)

    def __post_init__(self) -> None:
        if "source_file" not in self.metadata:
            self.metadata["source_file"] = "unknown"
        if "file_type" not in self.metadata:
            self.metadata["file_type"] = "unknown"


@dataclass
class ChunkResult:
    """Un chunk procesado, listo para almacenar."""

    chunk_id: str
    text: str
    embedding: Optional[np.ndarray]
    metadata: dict = field(default_factory=dict)

    # ── Campos de metadata esperados ────────────────────────────────
    # source_file  : str  — archivo de donde proviene
    # file_type    : str
    # chunk_index  : int  — índice del chunk dentro del documento
    # num_sentences: int  — cantidad de oraciones en el chunk
    # num_tokens   : int  — cantidad de tokens del chunk

    @staticmethod
    def generate_id() -> str:
        """Genera un ID único para el chunk."""
        return str(uuid.uuid4())


@dataclass
class SearchResult:
    """Resultado de una búsqueda en el almacén vectorial."""

    chunk_id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)
