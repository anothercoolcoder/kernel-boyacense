"""
Reader para archivos JSON genéricos.

Lee archivos JSON y extrae todo el contenido textual
de forma recursiva, concatenando los valores de tipo string.
"""

import json
import logging
from pathlib import Path

from extractor.models import Document
from extractor.readers.base import BaseReader

logger = logging.getLogger(__name__)


class JSONReader(BaseReader):
    """
    Lee archivos JSON y extrae texto de todos los valores string.

    Recorre recursivamente la estructura JSON (dicts, listas)
    y concatena todos los valores textuales encontrados.
    """

    def supported_extensions(self) -> set[str]:
        return {".json"}

    def _extract_texts(self, obj, path: str = "") -> list[tuple[str, str]]:
        """
        Extrae pares (ruta, texto) de una estructura JSON recursivamente.

        Returns
        -------
        list[tuple[str, str]]
            Lista de (json_path, text_value).
        """
        results: list[tuple[str, str]] = []

        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                results.extend(self._extract_texts(value, current_path))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                current_path = f"{path}[{i}]"
                results.extend(self._extract_texts(item, current_path))
        elif isinstance(obj, str) and obj.strip():
            results.append((path, obj.strip()))

        return results

    def read(self, file_path: Path) -> list[Document]:
        """
        Lee un archivo JSON y retorna documentos con el texto extraído.

        Parameters
        ----------
        file_path : Path
            Ruta al archivo JSON.

        Returns
        -------
        list[Document]
            Un Document con todo el texto extraído del JSON.
        """
        file_path = Path(file_path)
        logger.info("📋 Leyendo archivo JSON: %s", file_path.name)

        try:
            raw = file_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error("❌ Error al parsear JSON '%s': %s", file_path.name, e)
            raise

        text_pairs = self._extract_texts(data)

        if not text_pairs:
            logger.warning(
                "  └─ ⚠️  No se encontró texto en '%s'", file_path.name
            )
            return []

        # Concatenar todo el texto extraído
        full_text = "\n\n".join(text for _, text in text_pairs)

        metadata = {
            "source_file": str(file_path.resolve()),
            "file_type": ".json",
            "file_name": file_path.name,
            "content_type": "json_text",
            "text_fields_found": len(text_pairs),
        }

        logger.info(
            "  ├─ Campos textuales encontrados: %d",
            len(text_pairs),
        )
        logger.info(
            "  └─ Texto extraído: %d caracteres",
            len(full_text),
        )

        return [Document(text=full_text, metadata=metadata)]
