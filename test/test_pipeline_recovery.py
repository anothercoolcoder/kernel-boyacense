import tempfile
import unittest
from pathlib import Path

from lib.inventario_oficial import ResolucionAmbigua, resolver_archivo
from main import Checkpoint, CheckpointError


def _oficial(carpeta: str, nombre: str, numero: int) -> dict[str, str]:
    return {
        "DOC_ID": f"F1-CSET-{numero:03d}",
        "Carpeta": carpeta,
        "Nombre estandarizado": nombre,
        "Fenómeno": "F1",
        "Observatorio": "CSET",
        "Código Observatorio": "CSET",
        "Tipo": "informe",
    }


class TestInventarioSeguro(unittest.TestCase):
    def test_cset_ambiguo_conserva_seis_candidatos(self):
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "copiado" / "CSET_center-for-security-and-emerging-technology-2.pdf"
            ruta.parent.mkdir()
            ruta.write_bytes(b"pdf")
            inventario = [_oficial(f"oficial/{i}", ruta.name, i) for i in range(1, 7)]
            with self.assertRaises(ResolucionAmbigua) as contexto:
                resolver_archivo(ruta, inventario, Path(temporal) / "raiz-no-local")
            self.assertEqual(len(contexto.exception.candidatos), 6)
            self.assertIn("ambiguo", str(contexto.exception))

    def test_resuelve_por_sufijo_de_carpeta_en_raiz_alternativa(self):
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "copia" / "fenomeno_1" / "documento.pdf"
            ruta.parent.mkdir(parents=True)
            ruta.write_bytes(b"pdf")
            registro = _oficial("fenomeno_1", ruta.name, 1)
            resultado = resolver_archivo(ruta, [registro], Path(temporal) / "original")
            self.assertEqual(resultado["doc_id"], "F1-CSET-001")
            self.assertEqual(resultado["formato"], "pdf")


class TestCheckpoint(unittest.TestCase):
    def _crear(self, base: Path, nombre: str = "doc.txt") -> Path:
        ruta = base / nombre
        ruta.write_text("contenido", encoding="utf-8")
        return ruta

    def test_reanuda_sin_duplicar_y_tolera_linea_truncada(self):
        with tempfile.TemporaryDirectory() as temporal:
            base = Path(temporal)
            ruta = self._crear(base)
            checkpoint_path = base / "estado.jsonl"
            primero = Checkpoint(checkpoint_path, base, resume=False)
            registros = [{"doc_id": "DOC-1", "texto": "contenido"}]
            primero.persist_records(ruta, registros)
            primero.mark(ruta, "ok", 1)
            with checkpoint_path.open("a", encoding="utf-8") as archivo:
                archivo.write('{"kind":"document"')
            segundo = Checkpoint(checkpoint_path, base, resume=True)
            self.assertTrue(segundo.reusable(ruta))
            self.assertEqual(segundo.records[str(ruta.resolve())], registros)

    def test_archivo_modificado_no_es_reutilizable(self):
        with tempfile.TemporaryDirectory() as temporal:
            base = Path(temporal)
            ruta = self._crear(base)
            path = base / "estado.jsonl"
            checkpoint = Checkpoint(path, base, resume=False)
            checkpoint.persist_records(ruta, [{"texto": "viejo"}])
            checkpoint.mark(ruta, "ok", 1)
            ruta.write_text("contenido nuevo y distinto", encoding="utf-8")
            self.assertFalse(checkpoint.reusable(ruta))
            self.assertTrue(checkpoint.changed_since_checkpoint(ruta))

    def test_checkpoint_de_otro_corpus_se_rechaza(self):
        with tempfile.TemporaryDirectory() as temporal:
            base = Path(temporal)
            path = base / "estado.jsonl"
            Checkpoint(path, base / "corpus-a", resume=False)
            with self.assertRaises(CheckpointError):
                Checkpoint(path, base / "corpus-b", resume=True)

    def test_resume_sin_checkpoint_se_rechaza(self):
        with tempfile.TemporaryDirectory() as temporal:
            base = Path(temporal)
            with self.assertRaises(CheckpointError):
                Checkpoint(base / "inexistente.jsonl", base, resume=True)

    def test_running_queda_reintentable(self):
        with tempfile.TemporaryDirectory() as temporal:
            base = Path(temporal)
            ruta = self._crear(base)
            path = base / "estado.jsonl"
            checkpoint = Checkpoint(path, base, resume=False)
            checkpoint.mark(ruta, "running")
            recuperado = Checkpoint(path, base, resume=True)
            self.assertFalse(recuperado.reusable(ruta))


if __name__ == "__main__":
    unittest.main()
