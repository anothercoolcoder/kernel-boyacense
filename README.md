# Kernel Boyacense — Base de Conocimiento Vectorial RAG

Sistema de ingestión multimodal, fragmentación semántica, indexación vectorial con **FAISS** y visualización interactiva desarrollado para el **CODEFEST AD ASTRA 2026**.

---

## Tabla de Contenidos

1. [Requisitos Previos](#-requisitos-previos)
2. [Configuración del Entorno Virtual](#-configuración-del-entorno-virtual)
3. [Ejecución del Pipeline Principal (`main.py`)](#-ejecución-del-pipeline-principal-mainpy)
4. [Visualizador Web Interactivo (`visualizador_faiss.py`)](#-visualizador-web-interactivo-visualizador_faisspy)
5. [Herramienta CLI de Parsing y Exportación (`exportar_faiss.py`)](#-herramienta-cli-de-parsing-y-exportación-exportar_faisspy)
6. [Ejecución de Pruebas y Validación](#-ejecución-de-pruebas-y-validación)
7. [Estructura del Proyecto](#-estructura-del-proyecto)

---

## Requisitos Previos

- **Python:** Version 3.10 o superior.
- **Tesseract OCR (Opcional para procesamiento de imágenes):**
  - **Linux (Ubuntu/Debian):** `sudo apt update && sudo apt install -y tesseract-ocr tesseract-ocr-spa`
  - **macOS:** `brew install tesseract tesseract-lang`
  - **Windows:** Descargar instalador de [Tesseract OCR Wiki](https://github.com/UB-Mannheim/tesseract/wiki).

---

## Configuración del Entorno Virtual

Sigue estos pasos para crear e instalar las dependencias del proyecto:

### 1. Crear el entorno virtual
```bash
python3 -m venv venv
```

### 2. Activar el entorno virtual
- **En Linux / macOS:**
  ```bash
  source venv/bin/activate
  ```
- **En Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **En Windows (CMD):**
  ```cmd
  .\venv\Scripts\activate.bat
  ```

### 3. Actualizar `pip` e instalar dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Ejecución del Pipeline Principal (`main.py`)

El orquestador `main.py` ejecuta en secuencia las etapas de **Extracción Multimodal** → **Fragmentación** → **Indexación Vectorial (FAISS)**.

### Opciones de Ejecución

#### A. Procesar el corpus por defecto (`corpus_adl/`)
Ejecuta el pipeline completo extrayendo, fragmentando e indexando todos los archivos soportados en `corpus_adl/`:
```bash
python main.py
```

#### B. Procesar un directorio de corpus alternativo
Especifica una ruta personalizada como argumento:
```bash
python main.py ruta/a/tu_corpus
```

#### C. Modo `dry-run` (Validación rápida sin generar embeddings)
Útil para validar la extracción y fragmentación del corpus sin gastar tiempo en el cálculo de vectores/embeddings:
```bash
# Sobre el corpus por defecto:
python main.py --dry-run

# Sobre un corpus alternativo:
python main.py ruta/a/tu_corpus --dry-run
```

---

## Visualizador Web Interactivo (`visualizador_faiss.py`)

Proporciona un **Dashboard interactivo en tiempo real** para inspeccionar la base de conocimiento, explorar fragmentos (chunks) y auditar metadatos estructurados.

### Iniciar el Servidor Web

```bash
# Iniciar en el puerto por defecto (8501):
python visualizador_faiss.py

# O especificar un puerto personalizado (ej. 8080):
python visualizador_faiss.py 8080
```

Una vez iniciado, abre tu navegador en **`http://localhost:8501`**.

### Características de la Interfaz Web:
- **Dashboard & Métricas:** Estadísticas del índice (total de chunks, documentos únicos, conteo de tokens, distribución por formatos y documentos).
- ** Explorador de Chunks & Metadatos:** Tabla interactiva y filtros por texto, formato (`pdf`, `markdown`, `html`, `json`, `csv`, `imagen`, `osm_pbf`), fenómeno temático y documento. Muestra de forma explicita todos los campos de metadatos (`doc_id`, `chunk_id`, `fuente`, `formato`, `fenomeno`, `posicion`, `num_tokens` y `texto`) tanto en la tabla como en el visor modal detallado.
- ** Descarga de Datos:** Botones de un solo clic para exportar la base de datos parseada a JSON o CSV.

---

## Herramienta CLI de Parsing y Exportación (`exportar_faiss.py`)

Permite parsear el archivo `faiss_index/index.pkl` a varios formatos legibles desde la terminal o scripts automatizados.

### Comandos CLI Principales

```bash
# Exportar a todos los formatos (JSON, JSONL, CSV y Markdown) + mostrar resumen en consola:
python exportar_faiss.py --stats

# Exportar solo a un formato específico (opciones: all, json, jsonl, csv, md):
python exportar_faiss.py --format csv

# Realizar una búsqueda semántica de prueba desde la terminal:
python exportar_faiss.py --search "Inteligencia artificial militar" --top-k 5

# Especificar rutas personalizadas de índice y salida:
python exportar_faiss.py --faiss-dir ./faiss_index --out-dir ./salida
```

Los archivos generados se guardan automáticamente en la carpeta `salida/`:
- `salida/faiss_chunks.json`: Dataset estructurado completo en JSON.
- `salida/faiss_chunks.jsonl`: Dataset en formato JSON Lines.
- `salida/faiss_chunks.csv`: Archivo tabular para Excel / Pandas.
- `salida/faiss_resumen.md`: Reporte en Markdown con estadísticas y muestras.

---

## Ejecución de Pruebas y Validación

El proyecto cuenta con suites de prueba funcionales para validar la extracción, fragmentación e indexación vectorial:

### 1. Suite de Pruebas End-to-End (`test_pipeline.py`)
Ejecuta las autopruebas de los módulos de extracción/fragmentación y valida la coherencia de datos sobre archivos reales:
```bash
# Probar sobre todo el corpus en corpus_adl/:
python test_pipeline.py

# Probar sobre un archivo o lista de archivos específicos:
python test_pipeline.py corpus_adl/data/documento.pdf
```

### 2. Prueba de Recuperación e Índice Vectorial FAISS (`test/test_rag.py`)
Verifica la carga del índice vectorial `faiss_index/` y realiza consultas de prueba mediante similitud coseno con `multilingual-e5-large-instruct`:
```bash
python test/test_rag.py
```

### 3. Validación Rápida sin Embeddings (`main.py --dry-run`)
Valida la cadena completa de ingestión y fragmentación en memoria:
```bash
python main.py --dry-run
```

---

## Estructura del Proyecto

```text
kernel-boyacense/
├── context/               # Documentación técnica de referencia y reportes de diagnóstico
├── corpus_adl/            # Corpus de archivos de entrada (PDF, HTML, MD, CSV, imágenes, PBF)
├── extraccion/            # Módulo de extracción multimodal y fragmentación semántica
├── faiss_index/           # Índice vectorial FAISS persistido (index.faiss y index.pkl)
├── indexar/               # Módulo de creación e indexación vectorial con HuggingFace Embeddings
├── salida/                # Archivos exportados (JSON, JSONL, CSV, Markdown)
├── test/                  # Pruebas de recuperación semántica e integración RAG
├── test_pipeline.py       # Suite de pruebas end-to-end (extracción + fragmentación + metadatos)
├── main.py                # Orquestador principal del pipeline RAG
├── exportar_faiss.py      # Script de parsing, exportación y consultas CLI para FAISS
├── visualizador_faiss.py  # Servidor web y dashboard interactivo de visualización
├── requirements.txt       # Lista de dependencias del proyecto
└── README.md              # Documentación principal del proyecto
```
