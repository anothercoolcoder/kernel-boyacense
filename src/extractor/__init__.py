"""
Módulo extractor — Pipeline de extracción semántica.

Uso como módulo importable:

    from extractor import ExtractionPipeline
    pipeline = ExtractionPipeline()
    results = pipeline.process("/ruta/a/archivos/")
"""

from extractor.pipeline import ExtractionPipeline
from extractor.models import Document, ChunkResult

__all__ = ["ExtractionPipeline", "Document", "ChunkResult"]
