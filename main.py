"""Orquestador del pipeline RAG — kernel-boyacense.

Ejecuta las tres etapas en secuencia, pasando datos en memoria:

    extraccion  →  fragmentacion  →  indexacion (FAISS)

Uso:
    python main.py                         # procesa corpus_adl/ por defecto
    python main.py ruta/a/corpus           # carpeta alternativa
    python main.py ruta/a/corpus --dry-run # solo extrae + fragmenta, sin indexar

El flag --dry-run es útil para validar el corpus antes de gastar tiempo en
los embeddings.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# ── Logging ────────────────────────────────────────────────────────────────── #
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("main")

# ── Rutas base ─────────────────────────────────────────────────────────────── #
BASE_DIR   = Path(__file__).parent
CORPUS_ADL = BASE_DIR / "corpus_adl"
FAISS_DIR  = str(BASE_DIR / "faiss_index")

#: Mapeo de nombre de carpeta → número de fenómeno.
#: El pipeline infiere el fenómeno del primer ancestro del archivo que coincida.
FENOMENO_MAP: dict[str, int] = {
    "fenomeno_1": 1,
    "fenomeno_2": 2,
    "fenomeno_3": 3,
}


def _inferir_fenomeno(ruta: Path, texto: str = "") -> int:
    """Devuelve el número de fenómeno buscando ``fenomeno_N`` en el path o infiriendo por contenido."""
    from extraccion.fragmentacion import inferir_fenomeno
    return inferir_fenomeno(ruta, texto)


def _parse_args() -> tuple[Path, bool]:
    """Lee argumentos CLI mínimos sin dependencias externas."""
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    args = [a for a in args if a != "--dry-run"]
    corpus = Path(args[0]) if args else CORPUS_ADL
    return corpus, dry_run


def _descubrir_archivos(corpus: Path) -> list[Path]:
    """Lista todos los archivos en ``corpus`` (recursivo)."""
    from extraccion.extraccion import formatos_soportados

    soportados = set(formatos_soportados())
    archivos = sorted(
        p for p in corpus.rglob("*")
        if p.is_file()
        and not p.name.startswith(".")
        and p.suffix.lower() in soportados
    )
    return archivos


def main() -> None:
    corpus, dry_run = _parse_args()

    if not corpus.exists():
        logger.error("La carpeta de corpus no existe: %s", corpus)
        sys.exit(1)

    logger.info("Corpus: %s", corpus.resolve())
    logger.info("Modo:   %s", "dry-run (sin indexar)" if dry_run else "completo")

    # ── Etapa 1: Extracción ───────────────────────────────────────────────── #
    from extraccion.extraccion import extraer_documento, ErrorExtraccion

    archivos = _descubrir_archivos(corpus)
    if not archivos:
        logger.warning("No se encontraron archivos soportados en %s", corpus)
        sys.exit(0)

    logger.info("Archivos encontrados: %d", len(archivos))

    registros: list[dict] = []
    for ruta in archivos:
        try:
            nuevos = extraer_documento(ruta)
            # Inyectar fenomeno en la metadata de cada registro.
            fenomeno = _inferir_fenomeno(ruta, nuevos[0]["texto"] if nuevos else "")
            for r in nuevos:
                r.setdefault("metadata", {})["fenomeno"] = fenomeno
            registros.extend(nuevos)
            logger.info("  [+] %s → %d registros", ruta.name, len(nuevos))
        except ErrorExtraccion as exc:
            logger.warning("  [-] %s ignorado: %s", ruta.name, exc)

    logger.info("Extracción completa: %d registros totales", len(registros))

    if not registros:
        logger.warning("Sin registros con texto útil. Pipeline detenido.")
        sys.exit(0)

    # ── Etapa 2: Fragmentación ────────────────────────────────────────────── #
    from extraccion.fragmentacion import fragmentar_registros

    fragmentos = fragmentar_registros(registros)
    logger.info("Fragmentación completa: %d fragmentos", len(fragmentos))

    if dry_run:
        logger.info("--dry-run activo: indexación omitida.")
        max_tok = max((f["num_tokens"] for f in fragmentos), default=0)
        print(f"\nResumen dry-run: {len(registros)} registros → {len(fragmentos)} fragmentos")
        print(f"Tokens máximos por fragmento: {max_tok}")
        return

    # ── Etapa 3: Indexación ───────────────────────────────────────────────── #
    from indexar.indexar import indexar

    indexar(fragmentos, faiss_dir=FAISS_DIR)


if __name__ == "__main__":
    main()
