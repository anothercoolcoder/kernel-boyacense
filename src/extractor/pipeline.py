"""
Pipeline de extracción — orquesta readers → normalizer → chunker → embedder.

Provee logging detallado en cada paso para visibilidad del proceso.
"""

import logging
import time
from pathlib import Path

from extractor.chunker.sentence_chunker import SentenceChunker
from extractor.embedder.e5_embedder import E5Embedder
from extractor.models import ChunkResult, Document
from extractor.normalizer.text_normalizer import TextNormalizer
from extractor.readers.base import BaseReader
from extractor.readers.docling_reader import DoclingReader
from extractor.readers.json_reader import JSONReader
from extractor.readers.pbf_reader import PBFReader

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """
    Pipeline completo: lectura → normalización → chunking → embeddings.

    Parameters
    ----------
    max_tokens : int
        Tokens máximos por chunk. Default: 510.
    overlap_sentences : int
        Oraciones de solapamiento entre chunks. Default: 1.
    language : str
        Idioma para detección de oraciones. Default: "es".
    batch_size : int
        Batch size para generación de embeddings. Default: 32.
    device : str or None
        Dispositivo para el modelo. None = auto-detect.
    """

    def __init__(
        self,
        max_tokens: int = 510,
        overlap_sentences: int = 1,
        language: str = "es",
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        logger.info("=" * 60)
        logger.info("🚀 Inicializando ExtractionPipeline")
        logger.info("=" * 60)

        # Readers disponibles
        self._readers: list[BaseReader] = [
            JSONReader(),
            DoclingReader(),
            PBFReader(),
        ]

        # Componentes del pipeline
        self._normalizer = TextNormalizer()
        self._chunker = SentenceChunker(
            max_tokens=max_tokens,
            overlap_sentences=overlap_sentences,
            language=language,
        )
        self._embedder = E5Embedder(
            batch_size=batch_size,
            device=device,
        )

        logger.info("✅ Pipeline inicializado correctamente.")
        logger.info("=" * 60)

    def _get_reader(self, file_path: Path) -> BaseReader | None:
        """Selecciona el reader apropiado para un archivo."""
        for reader in self._readers:
            if reader.can_read(file_path):
                return reader
        return None

    def _get_supported_extensions(self) -> set[str]:
        """Retorna todas las extensiones soportadas por los readers."""
        extensions: set[str] = set()
        for reader in self._readers:
            extensions.update(reader.supported_extensions())
        return extensions

    def _discover_files(self, input_path: Path) -> list[Path]:
        """Descubre archivos soportados en un directorio."""
        supported = self._get_supported_extensions()
        files = [
            f for f in sorted(input_path.rglob("*"))
            if f.is_file() and f.suffix.lower() in supported
        ]
        return files

    def process_file(self, file_path: Path) -> list[ChunkResult]:
        """
        Procesa un solo archivo a través del pipeline completo.

        Parameters
        ----------
        file_path : Path
            Ruta al archivo a procesar.

        Returns
        -------
        list[ChunkResult]
            Chunks con embeddings listos para almacenar.
        """
        file_path = Path(file_path)
        start_time = time.time()

        logger.info("")
        logger.info("─" * 50)
        logger.info("📄 Procesando archivo: %s", file_path.name)
        logger.info("   Ruta: %s", file_path.resolve())
        logger.info("─" * 50)

        # ── Paso 1: Seleccionar reader ──────────────────────────────
        reader = self._get_reader(file_path)
        if reader is None:
            logger.error(
                "❌ No hay reader disponible para extensión '%s'",
                file_path.suffix,
            )
            logger.info(
                "   Extensiones soportadas: %s",
                ", ".join(sorted(self._get_supported_extensions())),
            )
            return []

        logger.info(
            "  [1/4] 📖 LECTURA — Reader: %s",
            type(reader).__name__,
        )
        t0 = time.time()
        documents = reader.read(file_path)
        logger.info(
            "  [1/4] ✅ Lectura completada — %d documento(s) en %.2fs",
            len(documents),
            time.time() - t0,
        )

        if not documents:
            logger.warning("  ⚠️  Sin contenido extraído. Saltando archivo.")
            return []

        # ── Paso 2: Normalización ───────────────────────────────────
        logger.info("  [2/4] 🔤 NORMALIZACIÓN — UTF-8/NFC + limpieza")
        t0 = time.time()
        for doc in documents:
            original_len = len(doc.text)
            doc.text = self._normalizer.normalize(doc.text)
            final_len = len(doc.text)
            if original_len != final_len:
                logger.info(
                    "        Texto normalizado: %d → %d chars (Δ%d)",
                    original_len,
                    final_len,
                    final_len - original_len,
                )
        logger.info(
            "  [2/4] ✅ Normalización completada en %.2fs",
            time.time() - t0,
        )

        # ── Paso 3: Chunking ───────────────────────────────────────
        logger.info("  [3/4] ✂️  CHUNKING — max %d tokens, respetando oraciones",
                     self._chunker.max_tokens)
        t0 = time.time()
        all_chunks: list[dict] = []
        chunk_metadata_list: list[dict] = []

        for doc_idx, doc in enumerate(documents):
            chunks = self._chunker.chunk(doc.text)
            for chunk_idx, chunk_data in enumerate(chunks):
                all_chunks.append(chunk_data)
                chunk_metadata_list.append({
                    **doc.metadata,
                    "chunk_index": chunk_idx,
                    "doc_index": doc_idx,
                    "num_sentences": chunk_data["num_sentences"],
                    "num_tokens": chunk_data["num_tokens"],
                })

        logger.info(
            "  [3/4] ✅ Chunking completado — %d chunks en %.2fs",
            len(all_chunks),
            time.time() - t0,
        )

        if not all_chunks:
            logger.warning("  ⚠️  Sin chunks generados. Saltando archivo.")
            return []

        # Mostrar resumen de chunks
        for i, chunk in enumerate(all_chunks):
            preview = chunk["text"][:80].replace("\n", " ")
            logger.info(
                "        Chunk %d: %d oraciones, %d tokens — «%s…»",
                i,
                chunk["num_sentences"],
                chunk["num_tokens"],
                preview,
            )

        # ── Paso 4: Embeddings ──────────────────────────────────────
        logger.info("  [4/4] 🧠 EMBEDDINGS — multilingual-e5-base (dim=768)")
        t0 = time.time()
        texts = [c["text"] for c in all_chunks]
        embeddings = self._embedder.embed_passages(texts)
        logger.info(
            "  [4/4] ✅ Embeddings generados — shape %s en %.2fs",
            embeddings.shape,
            time.time() - t0,
        )

        # ── Construir resultados ────────────────────────────────────
        results: list[ChunkResult] = []
        for i, (chunk_data, metadata) in enumerate(
            zip(all_chunks, chunk_metadata_list)
        ):
            results.append(ChunkResult(
                chunk_id=ChunkResult.generate_id(),
                text=chunk_data["text"],
                embedding=embeddings[i],
                metadata=metadata,
            ))

        elapsed = time.time() - start_time
        logger.info("")
        logger.info(
            "✅ Archivo '%s' procesado: %d chunks en %.2fs",
            file_path.name,
            len(results),
            elapsed,
        )

        return results

    def process_directory(self, dir_path: Path) -> list[ChunkResult]:
        """
        Procesa todos los archivos soportados en un directorio (recursivo).

        Parameters
        ----------
        dir_path : Path
            Ruta al directorio a procesar.

        Returns
        -------
        list[ChunkResult]
            Todos los chunks de todos los archivos.
        """
        dir_path = Path(dir_path)
        start_time = time.time()

        logger.info("")
        logger.info("=" * 60)
        logger.info("📁 Procesando directorio: %s", dir_path.resolve())
        logger.info("=" * 60)

        files = self._discover_files(dir_path)

        if not files:
            logger.warning(
                "⚠️  No se encontraron archivos soportados en '%s'",
                dir_path,
            )
            logger.info(
                "   Extensiones soportadas: %s",
                ", ".join(sorted(self._get_supported_extensions())),
            )
            return []

        logger.info(
            "📋 Archivos encontrados: %d", len(files),
        )
        for i, f in enumerate(files, start=1):
            logger.info("   %d. %s (%s)", i, f.name, f.suffix)

        all_results: list[ChunkResult] = []

        for file_idx, file_path in enumerate(files, start=1):
            logger.info("")
            logger.info(
                "━━━ Archivo %d/%d ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                file_idx,
                len(files),
            )
            try:
                results = self.process_file(file_path)
                all_results.extend(results)
            except Exception as e:
                logger.error(
                    "❌ Error procesando '%s': %s", file_path.name, e,
                )
                logger.info("   Continuando con el siguiente archivo…")

        elapsed = time.time() - start_time
        logger.info("")
        logger.info("=" * 60)
        logger.info("🏁 RESUMEN DEL DIRECTORIO")
        logger.info("=" * 60)
        logger.info("   Archivos procesados: %d/%d", len(files), len(files))
        logger.info("   Total chunks: %d", len(all_results))
        logger.info("   Tiempo total: %.2fs", elapsed)
        logger.info("=" * 60)

        return all_results

    def process(self, input_path: str | Path) -> list[ChunkResult]:
        """
        Punto de entrada flexible — acepta archivo o directorio.

        Parameters
        ----------
        input_path : str or Path
            Ruta a un archivo individual o un directorio.

        Returns
        -------
        list[ChunkResult]
            Chunks procesados.
        """
        input_path = Path(input_path)

        if input_path.is_dir():
            return self.process_directory(input_path)
        elif input_path.is_file():
            return self.process_file(input_path)
        else:
            logger.error("❌ La ruta no existe: %s", input_path)
            return []

    def process_files(self, file_paths: list[str | Path]) -> list[ChunkResult]:
        """
        Procesa una lista de archivos individuales.

        Parameters
        ----------
        file_paths : list[str | Path]
            Lista de rutas a archivos.

        Returns
        -------
        list[ChunkResult]
            Todos los chunks de todos los archivos.
        """
        start_time = time.time()

        logger.info("")
        logger.info("=" * 60)
        logger.info("📑 Procesando %d archivo(s) individualmente", len(file_paths))
        logger.info("=" * 60)

        all_results: list[ChunkResult] = []

        for file_idx, fp in enumerate(file_paths, start=1):
            fp = Path(fp)
            logger.info(
                "━━━ Archivo %d/%d ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                file_idx,
                len(file_paths),
            )
            try:
                results = self.process_file(fp)
                all_results.extend(results)
            except Exception as e:
                logger.error("❌ Error procesando '%s': %s", fp.name, e)
                logger.info("   Continuando con el siguiente archivo…")

        elapsed = time.time() - start_time
        logger.info("")
        logger.info("=" * 60)
        logger.info("🏁 RESUMEN")
        logger.info("=" * 60)
        logger.info("   Archivos procesados: %d", len(file_paths))
        logger.info("   Total chunks: %d", len(all_results))
        logger.info("   Tiempo total: %.2fs", elapsed)
        logger.info("=" * 60)

        return all_results
