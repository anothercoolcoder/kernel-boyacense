"""
CLI del extractor semántico.

Modos de uso:

    # Modo directorio — procesa todo lo que hay dentro
    python -m main --input ./datos/ --output ./indice/

    # Modo archivos individuales
    python -m main --files archivo1.pdf archivo2.html --output ./indice/
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Asegurar que src/ esté en el path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extractor.pipeline import ExtractionPipeline
from storage.faiss_store import FAISSStore


def _setup_logging(verbose: bool = False) -> None:
    """Configura logging con formato detallado."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Silenciar logs ruidosos de librerías externas
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("docling").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _build_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos CLI."""
    parser = argparse.ArgumentParser(
        prog="extractor",
        description=(
            "🔍 Extractor Semántico — Pipeline de extracción de texto, "
            "normalización, chunking, embeddings y almacenamiento en FAISS."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Procesar un directorio completo
  python main.py --input ./datos/ --output ./indice/

  # Procesar archivos individuales
  python main.py --files doc1.pdf doc2.html --output ./indice/

  # Con más detalle en el log
  python main.py --input ./datos/ --output ./indice/ --verbose
        """,
    )

    # Grupo mutuamente excluyente: --input XOR --files
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", "-i",
        type=str,
        help="Directorio o archivo individual a procesar.",
    )
    input_group.add_argument(
        "--files", "-f",
        nargs="+",
        type=str,
        help="Lista de archivos individuales a procesar.",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Directorio de salida para el índice FAISS y metadata.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=510,
        help="Máximo de tokens por chunk (default: 510).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=1,
        help="Oraciones de solapamiento entre chunks (default: 1).",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="es",
        choices=["es", "en", "pt"],
        help="Idioma para detección de oraciones (default: es).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size para embeddings (default: 32).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Muestra logs de nivel DEBUG.",
    )

    return parser


def main() -> None:
    """Punto de entrada principal del CLI."""
    parser = _build_parser()
    args = parser.parse_args()

    _setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║        🔍 EXTRACTOR SEMÁNTICO — KERNEL BOYACENSE       ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info("")

    start_time = time.time()

    # ── Inicializar pipeline ────────────────────────────────────────
    pipeline = ExtractionPipeline(
        max_tokens=args.max_tokens,
        overlap_sentences=args.overlap,
        language=args.language,
        batch_size=args.batch_size,
    )

    # ── Procesar según modo ─────────────────────────────────────────
    if args.input:
        input_path = Path(args.input)
        logger.info("📌 Modo: %s", "DIRECTORIO" if input_path.is_dir() else "ARCHIVO")
        results = pipeline.process(input_path)
    else:
        logger.info("📌 Modo: ARCHIVOS INDIVIDUALES (%d archivos)", len(args.files))
        file_paths = [Path(f) for f in args.files]

        # Validar que los archivos existen
        for fp in file_paths:
            if not fp.exists():
                logger.error("❌ Archivo no encontrado: %s", fp)
                sys.exit(1)

        results = pipeline.process_files(file_paths)

    # ── Almacenar en FAISS ──────────────────────────────────────────
    if results:
        logger.info("")
        logger.info("─" * 50)
        logger.info("💾 ALMACENAMIENTO EN FAISS")
        logger.info("─" * 50)

        store = FAISSStore(dimension=results[0].embedding.shape[0])
        store.add(results)

        output_dir = Path(args.output)
        store.save(output_dir)

        logger.info("")
        logger.info("✅ Índice guardado en: %s", output_dir.resolve())
    else:
        logger.warning("⚠️  No se generaron chunks. No se creó índice FAISS.")

    # ── Resumen final ───────────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║                    📊 RESUMEN FINAL                     ║")
    logger.info("╠══════════════════════════════════════════════════════════╣")
    logger.info("║  Chunks generados:  %-36d ║", len(results))
    if results:
        avg_tokens = sum(
            r.metadata.get("num_tokens", 0) for r in results
        ) / len(results)
        logger.info("║  Tokens promedio:   %-36.0f ║", avg_tokens)
        logger.info("║  Dimensión embeds:  %-36d ║", results[0].embedding.shape[0])
    logger.info("║  Tiempo total:      %-36s ║", f"{elapsed:.2f}s")
    logger.info("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
