"""Orquestador durable: extracción -> fragmentación -> FAISS.

El checkpoint es JSONL append-only. Los registros se escriben primero en un
archivo hermano y el estado ``ok`` después, por lo que un proceso muerto solo
puede repetir el documento en curso, nunca confirmar datos incompletos.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger("main")

BASE_DIR = Path(__file__).parent
CORPUS_ADL = BASE_DIR / "corpus_adl"
INVENTARIO_OFICIAL = BASE_DIR / "CORPUS CODEFEST AD ASTRA 2026" / "Indice_Datos_Codefest.xlsx"
if not INVENTARIO_OFICIAL.is_file() and (BASE_DIR / "Indice_Datos_Codefest.xlsx").is_file():
    INVENTARIO_OFICIAL = BASE_DIR / "Indice_Datos_Codefest.xlsx"
RAIZ_CORPUS_OFICIAL = BASE_DIR / "CORPUS CODEFEST AD ASTRA 2026"
FAISS_DIR = str(BASE_DIR / "base_vectorial" / "encoder_multilingual-e5-large-instruct")
FORMATOS_OFICIALES = (
    ".pdf", ".html", ".htm", ".xhtml", ".md", ".markdown", ".mdx", ".txt", ".text", ".log",
    ".json", ".csv", ".tsv", ".xlsx", ".xlsm", ".xls",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp",
    ".pbf", ".mvt", ".gpkg", ".osm.pbf"
)
PIPELINE_VERSION = "checkpoint-v1"


class CheckpointError(RuntimeError):
    pass


class InterrupcionPipeline(RuntimeError):
    pass


def _fingerprint(ruta: Path) -> dict[str, Any]:
    stat = ruta.stat()
    return {"ruta": str(ruta.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


class Checkpoint:
    def __init__(self, ruta: Path, corpus: Path, resume: bool, config: str = "default") -> None:
        self.ruta = ruta
        self.records_path = ruta.with_name(ruta.name + ".records.jsonl")
        self.corpus = str(corpus.resolve())
        self.config = config
        self.states: dict[str, dict[str, Any]] = {}
        self.records: dict[str, list[dict[str, Any]]] = {}
        self._load(resume)

    def _load(self, resume: bool) -> None:
        if self.ruta.exists() and not resume:
            raise CheckpointError(f"El checkpoint ya existe: {self.ruta}. Use --resume o elimínelo intencionalmente.")
        if not self.ruta.exists():
            if resume:
                raise CheckpointError(f"No existe el checkpoint solicitado para --resume: {self.ruta}")
            self._append({"kind": "header", "version": PIPELINE_VERSION, "corpus": self.corpus, "config": self.config, "created_at": time.time()})
            return
        lineas = self.ruta.read_text(encoding="utf-8").splitlines()
        for linea in lineas:
            try:
                evento = json.loads(linea)
            except json.JSONDecodeError:
                logger.warning("Última línea truncada del checkpoint ignorada: %s", self.ruta)
                break
            if evento.get("kind") == "header":
                if evento.get("version") != PIPELINE_VERSION or evento.get("corpus") != self.corpus or evento.get("config") != self.config:
                    raise CheckpointError("Checkpoint de otro corpus, versión o configuración; no se mezclará silenciosamente.")
            elif evento.get("kind") == "document":
                self.states[evento["ruta"]] = evento
        if self.records_path.exists():
            for linea in self.records_path.read_text(encoding="utf-8").splitlines():
                try:
                    evento = json.loads(linea)
                except json.JSONDecodeError:
                    logger.warning("Última línea truncada de registros ignorada: %s", self.records_path)
                    break
                self.records[evento["ruta"]] = evento.get("registros", [])

    def _append(self, evento: dict[str, Any]) -> None:
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        with self.ruta.open("a", encoding="utf-8") as archivo:
            archivo.write(json.dumps(evento, ensure_ascii=False, sort_keys=True) + "\n")
            archivo.flush()
            os.fsync(archivo.fileno())

    def mark(self, ruta: Path, estado: str, cantidad: int = 0, error: dict[str, Any] | None = None) -> None:
        evento = {"kind": "document", **_fingerprint(ruta), "estado": estado, "registros": cantidad, "error": error, "timestamp": time.time(), "pipeline": PIPELINE_VERSION}
        self._append(evento)
        self.states[evento["ruta"]] = evento

    def persist_records(self, ruta: Path, registros: list[dict[str, Any]]) -> None:
        self.records_path.parent.mkdir(parents=True, exist_ok=True)
        with self.records_path.open("a", encoding="utf-8") as archivo:
            archivo.write(json.dumps({"ruta": str(ruta.resolve()), "registros": registros}, ensure_ascii=False) + "\n")
            archivo.flush()
            os.fsync(archivo.fileno())
        self.records[str(ruta.resolve())] = registros

    def reusable(self, ruta: Path) -> bool:
        estado = self.states.get(str(ruta.resolve()))
        return bool(estado and estado.get("estado") in ("ok", "skipped") and estado.get("bytes") == ruta.stat().st_size and estado.get("mtime_ns") == ruta.stat().st_mtime_ns and str(ruta.resolve()) in self.records)

    def changed_since_checkpoint(self, ruta: Path) -> bool:
        estado = self.states.get(str(ruta.resolve()))
        return bool(estado and (estado.get("bytes") != ruta.stat().st_size or estado.get("mtime_ns") != ruta.stat().st_mtime_ns))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline durable de extracción documental")
    parser.add_argument("corpus", nargs="?", type=Path, default=CORPUS_ADL)
    parser.add_argument("--checkpoint", type=Path, default=BASE_DIR / "estado_pipeline.jsonl")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--faiss-dir", default=FAISS_DIR)
    return parser.parse_args()


def _descubrir_archivos(corpus: Path) -> list[Path]:
    return sorted(p for p in corpus.rglob("*") if p.is_file() and not p.name.startswith(".") and any(p.name.lower().endswith(ext) for ext in FORMATOS_OFICIALES))


def main() -> int:
    args = _parse_args()
    corpus = args.corpus
    if not corpus.is_dir():
        logger.error("La carpeta de corpus no existe: %s", corpus)
        return 1
    archivos = _descubrir_archivos(corpus)
    if not archivos:
        logger.warning("No se encontraron archivos soportados en %s", corpus)
        return 0
    if not INVENTARIO_OFICIAL.is_file():
        logger.error("Falta inventario oficial: %s", INVENTARIO_OFICIAL)
        return 1

    from lib.inventario_oficial import cargar_inventario, indexar_inventario, resolver_archivo
    from extraccion.extraccion import ErrorExtraccion, extraer_documento
    from extraccion.fragmentacion import fragmentar_registros

    config = json.dumps({"dry_run": args.dry_run, "faiss_dir": str(Path(args.faiss_dir).resolve()), "inventario": str(INVENTARIO_OFICIAL.resolve())}, sort_keys=True)
    try:
        checkpoint = Checkpoint(args.checkpoint, corpus, args.resume, config=config)
        inventario = cargar_inventario(INVENTARIO_OFICIAL)
        indexar_inventario(inventario)
    except (CheckpointError, OSError, ValueError) as exc:
        logger.error("Configuración o inventario inválido: %s", exc)
        return 1
    pendientes = [p for p in archivos if not checkpoint.reusable(p)]
    for ruta in pendientes:
        if checkpoint.changed_since_checkpoint(ruta):
            checkpoint.mark(ruta, "obsolete", error={"tipo": "ArchivoModificado", "mensaje": "cambió después del checkpoint", "etapa": "validacion"})
    recuperados = len(archivos) - len(pendientes)
    reintentos = sum(bool(checkpoint.states.get(str(p.resolve()))) for p in pendientes)
    logger.info("documentos totales=%d recuperados=%d pendientes=%d a_reintentar=%d", len(archivos), recuperados, len(pendientes), reintentos)

    interrumpido = False
    def detener(signum: int, _frame: Any) -> None:
        nonlocal interrumpido
        interrumpido = True
        logger.warning("Señal %s recibida; no se confirmará el documento actual.", signum)
    old_int = signal.signal(signal.SIGINT, detener)
    old_term = signal.signal(signal.SIGTERM, detener)
    inicio = time.monotonic()
    fragmentos: list[dict[str, Any]] = []
    resumen = {"ok": recuperados, "error": 0, "skipped": 0, "registros": 0}
    for numero, ruta in enumerate(archivos, 1):
        if checkpoint.reusable(ruta):
            registros_reusados = checkpoint.records[str(ruta.resolve())]
            if registros_reusados:
                fragmentos.extend(fragmentar_registros(registros_reusados))
            resumen["registros"] += len(registros_reusados)
            continue
        if interrumpido:
            break
        inicio_doc = time.monotonic()
        checkpoint.mark(ruta, "running")
        try:
            oficial = resolver_archivo(ruta, inventario, RAIZ_CORPUS_OFICIAL)
            registros = extraer_documento(ruta, metadata_oficial=oficial)
            if interrumpido:
                raise InterrupcionPipeline("interrumpido durante extracción")
            # La extracción queda recuperable incluso si fragmentación o
            # indexación fallan después.
            checkpoint.persist_records(ruta, registros)
            nuevos_fragmentos = fragmentar_registros(registros)
            if interrumpido:
                raise InterrupcionPipeline("interrumpido durante fragmentación")
            estado_final = "ok" if registros else "skipped"
            checkpoint.mark(ruta, estado_final, len(registros))
            fragmentos.extend(nuevos_fragmentos)
            resumen["ok"] += 1
            if not registros:
                resumen["skipped"] += 1
            resumen["registros"] += len(registros)
            logger.info("[+] %s [%s] -> %d registros (%.1fs)", ruta.name, oficial["doc_id"], len(registros), time.monotonic() - inicio_doc)
        except InterrupcionPipeline as exc:
            interrumpido = True
            try:
                checkpoint.mark(ruta, "interrupted", error={"tipo": type(exc).__name__, "mensaje": str(exc), "etapa": "señal"})
            except OSError:
                logger.exception("No se pudo persistir la interrupción de %s", ruta)
            break
        except (ErrorExtraccion, ValueError, OSError, MemoryError) as exc:
            error = {"tipo": type(exc).__name__, "mensaje": str(exc), "etapa": "extraccion_fragmentacion"}
            if hasattr(exc, "candidatos"):
                error["candidatos"] = getattr(exc, "candidatos")
                error["razon"] = getattr(exc, "razon", "")
            try:
                checkpoint.mark(ruta, "error", error=error)
            except OSError:
                logger.exception("No se pudo persistir el error de %s", ruta)
            resumen["error"] += 1
            logger.error("[-] %s: %s", ruta, error)
        if interrumpido:
            break
        procesados = numero
        porcentaje = 100 * procesados / len(archivos)
        transcurrido = time.monotonic() - inicio
        eta = (transcurrido / procesados) * (len(archivos) - procesados) if procesados else 0
        logger.info("Progreso %d/%d (%.1f%%), registros=%d, errores=%d, ETA=%.1fs, actual=%s", procesados, len(archivos), porcentaje, resumen["registros"], resumen["error"], eta, ruta.name)

    signal.signal(signal.SIGINT, old_int)
    signal.signal(signal.SIGTERM, old_term)
    if interrumpido:
        logger.error("Interrupción externa: checkpoint conservado; reanude con --resume")
        return 130
    if not fragmentos:
        logger.warning("Sin fragmentos útiles; no se reemplaza ningún índice")
    elif not args.dry_run:
        from indexar.indexar import indexar
        try:
            indexar(fragmentos, faiss_dir=args.faiss_dir)
        except (OSError, MemoryError, RuntimeError, ValueError) as exc:
            logger.exception("Indexación fallida; extracción permanece en checkpoint: %s", exc)
            return 1
    elapsed = time.monotonic() - inicio
    logger.info("Resumen: procesados=%d exitosos=%d omitidos=%d reintentables=%d registros=%d tiempo=%.1fs", len(archivos), resumen["ok"], resumen["skipped"], resumen["error"], resumen["registros"], elapsed)
    if args.dry_run:
        print(f"Resumen dry-run: {resumen['registros']} registros -> {len(fragmentos)} fragmentos")
    return 0 if resumen["error"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
