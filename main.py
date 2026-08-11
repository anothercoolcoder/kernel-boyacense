"""Orquestador del pipeline RAG — kernel-boyacense.

Ejecuta las tres etapas en secuencia, pasando datos en memoria:

    extraccion  →  fragmentacion  →  indexacion (FAISS)

Uso:
    python main.py                         # procesa corpus_adl/ por defecto
    python main.py ruta/a/corpus           # carpeta alternativa
    python main.py ruta/a/corpus --dry-run # solo extrae + fragmenta, sin indexar
    python main.py ruta/a/corpus --resume  # continúa desde checkpoint

El flag --dry-run es útil para validar el corpus antes de gastar tiempo en
los embeddings.
"""

from __future__ import annotations

import json
import logging
import sys
import time
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
INVENTARIO_OFICIAL = BASE_DIR / "CORPUS CODEFEST AD ASTRA 2026" / "Indice_Datos_Codefest.xlsx"
RAIZ_CORPUS_OFICIAL = BASE_DIR / "CORPUS CODEFEST AD ASTRA 2026"
FAISS_DIR  = str(BASE_DIR / "base_vectorial" / "encoder_multilingual-e5-large-instruct")

#: Mapeo de nombre de carpeta → número de fenómeno.
#: El pipeline infiere el fenómeno del primer ancestro del archivo que coincida.
FENOMENO_MAP: dict[str, int] = {
    "fenomeno_1": 1,
    "fenomeno_2": 2,
    "fenomeno_3": 3,
}
FORMATOS_OFICIALES = (".pdf", ".json", ".csv", ".xlsx", ".jpg", ".avif", ".txt", ".pbf")


def _inferir_fenomeno(ruta: Path, texto: str = "") -> int:
    """Devuelve el número de fenómeno buscando ``fenomeno_N`` en el path o infiriendo por contenido."""
    from extraccion.fragmentacion import inferir_fenomeno
    return inferir_fenomeno(ruta, texto)


def _parse_args() -> tuple[Path, bool, str, Path, bool]:
    """Lee argumentos CLI mínimos sin dependencias externas."""
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    resume = "--resume" in args
    args = [a for a in args if a != "--dry-run"]
    args = [a for a in args if a != "--resume"]
    faiss_dir = FAISS_DIR
    checkpoint = BASE_DIR / "checkpoint_extraccion.jsonl"
    if "--faiss-dir" in args:
        indice = args.index("--faiss-dir")
        if indice + 1 >= len(args):
            raise ValueError("--faiss-dir requiere una ruta")
        faiss_dir = args[indice + 1]
        args = args[:indice] + args[indice + 2:]
    if "--checkpoint" in args:
        indice = args.index("--checkpoint")
        if indice + 1 >= len(args):
            raise ValueError("--checkpoint requiere una ruta")
        checkpoint = Path(args[indice + 1])
        args = args[:indice] + args[indice + 2:]
    corpus = Path(args[0]) if args else CORPUS_ADL
    return corpus, dry_run, faiss_dir, checkpoint, resume


def _cargar_checkpoint(ruta: Path) -> dict[str, list[dict]]:
    """Carga éxitos; ignora última línea truncada por un corte abrupto."""
    recuperados: dict[str, list[dict]] = {}
    if not ruta.is_file():
        return recuperados
    with ruta.open(encoding="utf-8") as archivo:
        for linea in archivo:
            try:
                evento = json.loads(linea)
            except json.JSONDecodeError:
                continue
            if evento.get("estado") == "ok" and evento.get("ruta"):
                recuperados[evento["ruta"]] = evento.get("registros", [])
    return recuperados


def _guardar_checkpoint(ruta: Path, evento: dict) -> None:
    """Persiste un documento antes de continuar con el siguiente."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("a", encoding="utf-8") as archivo:
        archivo.write(json.dumps(evento, ensure_ascii=False) + "\n")
        archivo.flush()
        import os
        os.fsync(archivo.fileno())


def _descubrir_archivos(corpus: Path) -> list[Path]:
    """Lista todos los archivos en ``corpus`` (recursivo)."""
    archivos = sorted(
        p for p in corpus.rglob("*")
        if p.is_file()
        and not p.name.startswith(".")
        and any(p.name.lower().endswith(ext) for ext in FORMATOS_OFICIALES)
    )
    return archivos


def _duracion(segundos: float) -> str:
    """Formatea segundos para progreso humano, sin dependencia externa."""
    segundos = max(0, int(segundos))
    horas, resto = divmod(segundos, 3600)
    minutos, segundos = divmod(resto, 60)
    if horas:
        return f"{horas}h {minutos:02d}m"
    return f"{minutos}m {segundos:02d}s"


def _mostrar_progreso(completados: int, total: int, registros: int, errores: int, inicio: float) -> None:
    """Dibuja avance documental y ETA; funciona también sin TTY."""
    transcurrido = time.monotonic() - inicio
    proporcion = completados / total if total else 1.0
    ancho = 28
    llenos = int(ancho * proporcion)
    barra = "#" * llenos + "." * (ancho - llenos)
    if completados:
        restante = transcurrido / completados * (total - completados)
        eta = _duracion(restante)
    else:
        eta = "--"
    linea = (
        f"Documentos [{barra}] {completados}/{total} ({proporcion:6.1%}) "
        f"registros={registros} errores={errores} "
        f"transcurrido={_duracion(transcurrido)} ETA={eta}"
    )
    if sys.stderr.isatty():
        sys.stderr.write("\r\033[K" + linea)
        sys.stderr.flush()
    elif completados == total or completados % 25 == 0:
        sys.stderr.write(linea + "\n")
        sys.stderr.flush()


def main() -> None:
    inicio_total = time.monotonic()
    corpus, dry_run, faiss_dir, checkpoint, resume = _parse_args()

    if not corpus.exists():
        logger.error("La carpeta de corpus no existe: %s", corpus)
        sys.exit(1)

    logger.info("Corpus: %s", corpus.resolve())
    logger.info("Modo:   %s", "dry-run (sin indexar)" if dry_run else "completo")

    from lib.inventario_oficial import cargar_inventario, resolver_archivo
    if not INVENTARIO_OFICIAL.is_file():
        logger.error("Falta inventario oficial: %s", INVENTARIO_OFICIAL)
        sys.exit(1)
    inventario = cargar_inventario(INVENTARIO_OFICIAL)

    # ── Etapa 1: Extracción ───────────────────────────────────────────────── #
    from extraccion.extraccion import extraer_documento, ErrorExtraccion

    archivos = _descubrir_archivos(corpus)
    if not archivos:
        logger.warning("No se encontraron archivos soportados en %s", corpus)
        sys.exit(0)

    logger.info("Archivos encontrados: %d", len(archivos))

    registros: list[dict] = []
    omitidos: list[tuple[str, str]] = []
    procesados: dict[str, list[dict]] = _cargar_checkpoint(checkpoint) if resume else {}
    if resume:
        registros.extend(
            registro
            for ruta in archivos
            for registro in procesados.get(str(ruta.resolve()), [])
        )
        logger.info("Reanudación: %d documentos recuperados desde %s", len(procesados), checkpoint)
    else:
        checkpoint.unlink(missing_ok=True)
    inicio_extraccion = time.monotonic()
    _mostrar_progreso(len(procesados), len(archivos), len(registros), 0, inicio_extraccion)
    for numero, ruta in enumerate(archivos, start=1):
        clave_ruta = str(ruta.resolve())
        if clave_ruta in procesados:
            continue
        try:
            metadata_oficial = resolver_archivo(ruta, inventario, RAIZ_CORPUS_OFICIAL)
            nuevos = extraer_documento(ruta, metadata_oficial=metadata_oficial)
            registros.extend(nuevos)
            procesados[clave_ruta] = nuevos
            _guardar_checkpoint(checkpoint, {
                "ruta": clave_ruta,
                "estado": "ok",
                "registros": nuevos,
            })
            logger.info("  [+] %s [%s] -> %d registros", ruta.name, metadata_oficial["doc_id"], len(nuevos))
        except (ErrorExtraccion, ValueError) as exc:
            # Un nombre duplicado en inventario no debe abortar 1.800 documentos.
            # Sin DOC_ID inequívoco se omite: inventar metadata rompe auditoría.
            omitidos.append((ruta.name, str(exc)))
            _guardar_checkpoint(checkpoint, {
                "ruta": clave_ruta,
                "estado": "error",
                "error": str(exc),
            })
            logger.warning("  [-] %s ignorado: %s", ruta.name, exc)
        finally:
            _mostrar_progreso(numero, len(archivos), len(registros), len(omitidos), inicio_extraccion)

    if sys.stderr.isatty():
        sys.stderr.write("\n")
        sys.stderr.flush()

    logger.info("Extracción completa: %d registros totales", len(registros))
    if omitidos:
        logger.warning("Documentos omitidos: %d (errores no fatales)", len(omitidos))
    logger.info("Tiempo extracción: %s", _duracion(time.monotonic() - inicio_extraccion))

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
        logger.info("Tiempo total: %s", _duracion(time.monotonic() - inicio_total))
        return

    # ── Etapa 3: Indexación ───────────────────────────────────────────────── #
    from indexar.indexar import indexar

    indexar(fragmentos, faiss_dir=faiss_dir)
    logger.info("Tiempo total: %s", _duracion(time.monotonic() - inicio_total))


if __name__ == "__main__":
    main()
