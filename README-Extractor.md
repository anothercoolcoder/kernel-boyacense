# Buscador Semantico - Extractor (Fase 1)

Este proyecto contiene el componente extractor para el buscador semantico, desarrollado para la hackaton. El modulo esta diseñado para procesar archivos de multiples formatos, limpiar y normalizar el texto, segmentar el texto en chunks sin romper oraciones, generar embeddings vectoriales con un modelo multilingue tipo BERT y almacenar los resultados de forma provisional en FAISS.

## Estructura del Modulo

El codigo se encuentra organizado bajo la carpeta `src/` con la siguiente estructura:

* `src/main.py`: Punto de entrada de la interfaz de linea de comandos (CLI).
* `src/extractor/`: Paquete principal del extractor.
  * `readers/`: Modulo encargado de la lectura y extraccion de texto de archivos (PDF, HTML, DOCX, PPTX, XLSX, CSV, MD, JSON, IMG, PBF).
  * `normalizer/`: Modulo que realiza la limpieza de texto y normalizacion Unicode UTF-8 (NFC).
  * `chunker/`: Modulo que divide el texto en segmentos (chunks) respetando los limites de oraciones completas y controlando el limite de tokens.
  * `embedder/`: Modulo que genera los vectores dense a partir de los chunks utilizando un encoder multilingue.
  * `pipeline.py`: Clase orquestadora que une los componentes anteriores y ofrece logs del proceso.
* `src/storage/`: Modulo para la persistencia provisional en FAISS.
* `src/tests/fixtures/`: Archivos de muestra en diferentes formatos e idiomas para validar el pipeline.

---

## Requisitos e Instalacion

El proyecto requiere Python (probado en version 3.14.6) y las siguientes dependencias de codigo abierto:

```bash
pip install docling osmium pysbd sentence-transformers faiss-cpu numpy transformers
```

---

## Guia de Uso

El extractor se puede ejecutar de dos formas principales: mediante la consola (CLI) o importando el pipeline directamente en codigo Python.

### 1. Interfaz de Linea de Comandos (CLI)

El CLI ofrece dos modos de entrada (directorio o lista de archivos) y permite parametrizar las opciones de chunking y logs.

#### Modo Directorio
Procesa todos los archivos con extensiones compatibles dentro del directorio especificado (busqueda recursiva):

```bash
python main.py --input ./tests/fixtures/ --output ./tests/output/
```

#### Modo Archivos Individuales
Procesa unicamente la lista de archivos provistos por parametro:

```bash
python main.py --files ./tests/fixtures/sample_boyaca.md ./tests/fixtures/patrimonio_colombia.html --output ./tests/output_files/
```

#### Opciones del CLI
* `--input`, `-i`: Ruta del directorio o archivo unico a procesar.
* `--files`, `-f`: Lista de archivos individuales separados por espacios.
* `--output`, `-o`: Directorio donde se guardaran el indice FAISS (`index.faiss`) y su metadata (`metadata.json`).
* `--max-tokens`: Numero maximo de tokens por chunk (por defecto 510).
* `--overlap`: Cantidad de oraciones de solapamiento entre chunks continuos (por defecto 1).
* `--language`: Idioma de deteccion de oraciones para pysbd (opciones: `es`, `en`, `pt`; por defecto `es`).
* `--batch-size`: Tamaño del lote para procesar embeddings (por defecto 32).
* `--verbose`, `-v`: Habilita logs detallados de nivel DEBUG.

### 2. Uso como Modulo de Python

El extractor esta expuesto como una API modular. Puede importarse en otros componentes del buscador semantico:

```python
from pathlib import Path
from extractor import ExtractionPipeline

# Inicializar el pipeline con parametros personalizados
pipeline = ExtractionPipeline(
    max_tokens=510,
    overlap_sentences=1,
    language="es"
)

# Procesar un directorio completo
results = pipeline.process(Path("./tests/fixtures/"))

# Procesar una lista de archivos
# results = pipeline.process_files([Path("doc1.pdf"), Path("doc2.csv")])

# Acceder a los resultados obtenidos
for r in results:
    print(f"ID: {r.chunk_id}")
    print(f"Texto: {r.text[:100]}...")
    print(f"Embedding shape: {r.embedding.shape}")  # (768,)
    print(f"Metadata: {r.metadata}")
```

Para realizar busquedas o cargar el indice en memoria utilizando el modulo de almacenamiento:

```python
from storage.faiss_store import FAISSStore
from extractor.embedder.e5_embedder import E5Embedder

# Cargar el almacenamiento vectorial
store = FAISSStore(dimension=768)
store.load("./tests/output/")

# Generar embedding de consulta (query)
embedder = E5Embedder()
query_vector = embedder.embed_query("municipios de Boyaca con aguas termales")

# Buscar top 3 resultados mas similares (Cosine Similarity)
search_results = store.search(query_vector, k=3)
for res in search_results:
    print(f"Score: {res.score:.4f} | Chunk ID: {res.chunk_id}")
    print(f"Texto: {res.text}\n")
```

---

## Librerias Utilizadas y Justificacion

1. **Docling**: Utilizada para la extraccion de multiples formatos de oficina (PDF, HTML, DOCX, XLSX, CSV, Markdown, e imagenes via OCR interno). Permite unificar los formatos de entrada en una estructura intermedia y soporta la linealizacion de tablas de manera automatica.
2. **Osmium (pyosmium)**: Utilizada para parsear archivos binarios PBF de OpenStreetMap. Extrae campos de texto como nombres, descripciones y direcciones de nodos, caminos y relaciones geograficas.
3. **PySBD (Python Sentence Boundary Disambiguation)**: Implementacion en Python del Pragmatic Segmenter. Es una libreria basada en reglas sumamente precisa para detectar los limites de oraciones en textos ruidosos (como los extraidos de PDFs), evitando romper abreviaturas o numeros flotantes.
4. **Sentence-Transformers**: Utilizada para cargar el modelo de embeddings. Evita el uso de APIs comerciales y permite la ejecucion local.
5. **Transformers (HuggingFace)**: Utilizado el tokenizador para contar con precision los tokens reales de cada oracion y asegurar que los chunks respeten exactamente la capacidad del encoder.
6. **FAISS (CPU)**: Libreria de busqueda vectorial de Meta. Utilizada para almacenar y buscar vectores en memoria o disco de forma eficiente mediante un indice IndexFlatIP (Inner Product).

---

## Errores y Limitaciones Documentados

### Error de Conversion de JSON en Docling
* **Sintoma**: Docling arrojaba un error de conversion fallida (`Conversion failed for: ... with status: failure. Errors: The document backend could not parse the input`) al intentar leer archivos `.json`.
* **Causa**: Docling espera un formato JSON propietario especifico de su esquema interno (DocLang) y no esta diseñado para leer estructuras JSON genericas de datos arbitrarios.
* **Solucion**: Se creo el lector custom `JSONReader` para interceptar archivos `.json` antes de que pasen por el lector de Docling. Este lector recorre de forma recursiva cualquier objeto o lista JSON y extrae todos los valores de tipo cadena de texto.

### Advertencia de Metodo Deprecado en Sentence-Transformers
* **Sintoma**: Aparecia una advertencia de tipo `FutureWarning: The get_sentence_embedding_dimension method has been renamed to get_embedding_dimension`.
* **Causa**: Cambios internos en las versiones recientes de la libreria `sentence-transformers`.
* **Solucion**: Se modificaron las llamadas en el modulo `E5Embedder` para invocar el metodo correcto `get_embedding_dimension()`.

### Limitacion del Tamaño de Oraciones
* **Sintoma**: Si una oracion individual extraida de un documento excede el limite maximo de tokens permitido por chunk (por ejemplo, textos sin puntuar o tablas muy grandes linealizadas como una sola linea), el chunker la mantendra completa violando la regla del max_tokens.
* **Causa**: El requisito del proyecto exige explicitamente que ninguna oracion sea dividida en diferentes chunks.
* **Solucion**: El componente `SentenceChunker` detecta este escenario, emite una advertencia en el log del pipeline y coloca la oracion entera en un chunk individual sin fraccionarla, priorizando la integridad semantica de la oracion sobre el limite de tokens.

### Requisito de Prefijo en Modelo E5
* **Sintoma**: Perdida de precision en las busquedas semanticas si no se configuran correctamente las entradas.
* **Causa**: La familia de modelos `multilingual-e5` requiere que el texto de los documentos almacenados tenga el prefijo `"passage: "` y las consultas de busqueda el prefijo `"query: "`.
* **Solucion**: El embedder implementa estos prefijos de manera transparente en sus metodos `embed_passages` and `embed_query`.
