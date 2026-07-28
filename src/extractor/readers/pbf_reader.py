"""
Reader para archivos PBF de OpenStreetMap.

Utiliza la librería osmium para leer archivos .osm.pbf
y extraer tags textuales de nodos, ways y relaciones.
"""

import logging
from pathlib import Path

import osmium

from extractor.models import Document
from extractor.readers.base import BaseReader

logger = logging.getLogger(__name__)

# Tags que contienen información textual relevante
_TEXT_TAGS: set[str] = {
    "name", "name:es", "name:en", "name:pt",
    "description", "note", "comment",
    "addr:street", "addr:city", "addr:state", "addr:country",
    "place", "amenity", "shop", "tourism", "historic",
    "natural", "landuse", "building", "highway",
    "operator", "brand", "website",
}


class _OSMHandler(osmium.SimpleHandler):
    """Handler interno que recolecta textos de elementos OSM."""

    def __init__(self) -> None:
        super().__init__()
        self.entries: list[dict] = []
        self.counts = {"nodes": 0, "ways": 0, "relations": 0}

    def _extract_tags(self, osm_object, obj_type: str) -> None:
        """Extrae tags textuales de un elemento OSM."""
        text_parts = []
        tags_found = {}

        for tag in osm_object.tags:
            if tag.k in _TEXT_TAGS:
                text_parts.append(f"{tag.k}: {tag.v}")
                tags_found[tag.k] = tag.v

        if text_parts:
            self.entries.append({
                "text": ". ".join(text_parts),
                "osm_type": obj_type,
                "osm_id": osm_object.id,
                "tags": tags_found,
            })
            self.counts[f"{obj_type}s"] += 1

    def node(self, n) -> None:
        self._extract_tags(n, "node")

    def way(self, w) -> None:
        self._extract_tags(w, "way")

    def relation(self, r) -> None:
        self._extract_tags(r, "relation")


class PBFReader(BaseReader):
    """
    Lee archivos .osm.pbf y extrae texto de los tags OSM.

    Agrupa las entradas en un solo Document por archivo,
    concatenando todos los textos extraídos.
    """

    def supported_extensions(self) -> set[str]:
        return {".pbf"}

    def read(self, file_path: Path) -> list[Document]:
        """
        Lee un archivo PBF y retorna documentos con el texto de los tags.

        Parameters
        ----------
        file_path : Path
            Ruta al archivo .osm.pbf.

        Returns
        -------
        list[Document]
            Un Document con todo el texto extraído del archivo.
        """
        file_path = Path(file_path)
        logger.info("🗺️  Leyendo archivo PBF: %s", file_path.name)

        handler = _OSMHandler()

        try:
            handler.apply_file(str(file_path))
        except Exception as e:
            logger.error("❌ Error al leer PBF '%s': %s", file_path.name, e)
            raise

        if not handler.entries:
            logger.warning(
                "  └─ ⚠️  No se encontraron tags textuales en '%s'",
                file_path.name,
            )
            return []

        # Concatenar todos los textos extraídos
        full_text = "\n\n".join(entry["text"] for entry in handler.entries)

        metadata = {
            "source_file": str(file_path.resolve()),
            "file_type": ".pbf",
            "file_name": file_path.name,
            "content_type": "osm_data",
            "osm_stats": handler.counts,
            "total_entries": len(handler.entries),
        }

        logger.info(
            "  ├─ Elementos encontrados — nodos: %d, ways: %d, relaciones: %d",
            handler.counts["nodes"],
            handler.counts["ways"],
            handler.counts["relations"],
        )
        logger.info(
            "  └─ Total entradas textuales: %d (%d caracteres)",
            len(handler.entries),
            len(full_text),
        )

        return [Document(text=full_text, metadata=metadata)]
