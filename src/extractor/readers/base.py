"""
Clase abstracta base para todos los readers.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from extractor.models import Document


class BaseReader(ABC):
    """Interfaz base que todo reader debe implementar."""

    @abstractmethod
    def read(self, file_path: Path) -> list[Document]:
        """
        Lee un archivo y retorna una lista de Document.

        Cada Document contiene el texto extraído y metadatos
        del archivo fuente (nombre, tipo, página, etc.).

        Parameters
        ----------
        file_path : Path
            Ruta al archivo a leer.

        Returns
        -------
        list[Document]
            Lista de documentos extraídos.
        """
        ...

    @abstractmethod
    def supported_extensions(self) -> set[str]:
        """
        Retorna el conjunto de extensiones soportadas por este reader.

        Returns
        -------
        set[str]
            Extensiones con punto, e.g. {".pdf", ".html"}.
        """
        ...

    def can_read(self, file_path: Path) -> bool:
        """Verifica si este reader puede leer el archivo dado."""
        return file_path.suffix.lower() in self.supported_extensions()
