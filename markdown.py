from pathlib import Path
from docling.document_converter import DocumentConverter

source = "/home/anothercoolcoder/principito.pdf"  # archivo o enlace

converter = DocumentConverter()
result = converter.convert(source)
doc = result.document

contenido_md = doc.export_to_markdown()

salida_path = Path("salida.md")
salida_path.write_text(contenido_md,encoding="utf-8")

print(f"Documento guardado con exito  en {salida_path.resolve()}")
