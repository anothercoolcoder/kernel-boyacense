"""
Reader basado en Docling para la mayoría de formatos de documentos.

Soporta: PDF, HTML, DOCX, PPTX, XLSX, CSV, Markdown, imágenes (PNG, JPG, JPEG).
"""

import logging
from pathlib import Path

from docling.document_converter import DocumentConverter

from extractor.models import Document
from extractor.readers.base import BaseReader

logger = logging.getLogger(__name__)

# Extensiones soportadas por Docling que nos interesan
_DOCLING_EXTENSIONS: set[str] = {
    ".pdf",
    ".html", ".htm",
    ".docx",
    ".pptx",
    ".xlsx",
    ".csv",
    ".md", ".markdown",
    ".png", ".jpg", ".jpeg",
}


class DoclingReader(BaseReader):
    """
    Lee documentos usando la librería Docling.

    Extrae texto y tablas; las tablas se linealizan como texto
    separado por pipe para que el chunker las pueda procesar.
    """

    def __init__(self) -> None:
        self._converter = DocumentConverter()

    def supported_extensions(self) -> set[str]:
        return _DOCLING_EXTENSIONS

    def read(self, file_path: Path) -> list[Document]:
        """
        Lee un archivo con Docling y retorna documentos extraídos.

        Parameters
        ----------
        file_path : Path
            Ruta al archivo a procesar.

        Returns
        -------
        list[Document]
            Un Document por cada sección lógica del archivo
            (párrafos + tablas linealizadas).
        """
        file_path = Path(file_path)
        logger.info("📖 Leyendo archivo con Docling: %s", file_path.name)

        try:
            result = self._converter.convert(str(file_path))
        except Exception as e:
            logger.error("❌ Error al convertir '%s': %s", file_path.name, e)
            raise

        doc = result.document
        base_metadata = {
            "source_file": str(file_path.resolve()),
            "file_type": file_path.suffix.lower(),
            "file_name": file_path.name,
        }

        documents: list[Document] = []

        # ── Extraer texto principal ─────────────────────────────────
        texto_md = doc.export_to_markdown()
        paragraphs = [
            p.replace("\t", " ").strip()
            for p in texto_md.split("\n\n")
            if p.strip()
        ]

        if paragraphs:
            full_text = "\n\n".join(paragraphs)
            documents.append(Document(
                text=full_text,
                metadata={**base_metadata, "content_type": "text"},
            ))
            logger.info(
                "  ├─ Texto extraído: %d párrafos, %d caracteres",
                len(paragraphs),
                len(full_text),
            )

        # ── Extraer tablas (linearizadas) ───────────────────────────
        for idx, table in enumerate(doc.tables, start=1):
            try:
                df = table.export_to_dataframe(doc)
                # Linealizar: cabecera + filas separadas por pipe
                header = " | ".join(str(c) for c in df.columns)
                rows = []
                for _, row in df.iterrows():
                    rows.append(" | ".join(str(v) for v in row.values))
                table_text = header + "\n" + "\n".join(rows)

                documents.append(Document(
                    text=table_text,
                    metadata={
                        **base_metadata,
                        "content_type": "table",
                        "table_index": idx,
                    },
                ))
                logger.info(
                    "  ├─ Tabla %d extraída: %d filas × %d columnas",
                    idx,
                    len(df),
                    len(df.columns),
                )
            except Exception as e:
                logger.warning(
                    "  ├─ ⚠️  No se pudo extraer tabla %d: %s", idx, e
                )

        if not documents:
            logger.warning(
                "  └─ ⚠️  No se extrajo contenido de '%s'", file_path.name
            )

        logger.info(
            "  └─ Total documentos extraídos de '%s': %d",
            file_path.name,
            len(documents),
        )
        return documents
