from pathlib import Path
from docling.document_converter import DocumentConverter,InputFormat

input_dir = Path("./documentos_entrada")
output_dir = Path(".documentos_salida")

output_dir.mkdir(parents=True, exist_ok=True)

converter = DocumentConverter()

extensiones_soportadas = {
    ext 
    for fmt in InputFormat
    for ext in fmt.supported_file_extensions
}

archivos = [
    f for f in input_dir.iterdir()
    if f.is_file() and f.suffix.lower() in extensiones_soportadas
]

print(f"Se encontraron {len(archivos)} archivos compatibles con Docling.\n")


# converter = DocumentConverter()
# result = converter.convert(source)
# doc = result.document

# parrafos = [item.text for item in doc.texts]

# tablas = []

# for table in doc.tables:
    # df = table.export_to_dataframe()
    # tablas.append(df.values.tolist())

# print(f"Total de párrafos extraídos: {len(parrafos)}")
# print("\nPrimeros 3 párrafos:")
# print(parrafos[:3])

# if tablas:
    # print("\nPrimera tabla extraída como arreglo de filas:")
    # print(tablas[0])