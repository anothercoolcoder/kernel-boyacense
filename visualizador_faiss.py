#!/usr/bin/env python3
"""Servidor Web Interactivo y Visualizador de Metadatos y Chunks para el Índice FAISS del Kernel Boyacense.

Proporciona un Dashboard interactivo, exploración estructurada de metadatos y exportación de datos.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("visualizador")

FAISS_DIR = Path(__file__).parent / "base_vectorial" / "encoder_multilingual-e5-large-instruct"
PORT = 8501

# Variables globales para cache en memoria
INDEX_DATA: List[Dict[str, Any]] = []
RESUMEN_STATS: Dict[str, Any] = {}


def cargar_indice_y_metadatos():
    """Carga los chunks y metadatos desde metadata.jsonl."""
    global INDEX_DATA, RESUMEN_STATS
    from exportar_faiss import cargar_datos_faiss, generar_resumen_estadistico

    logger.info("Cargando metadatos FAISS en memoria...")
    INDEX_DATA = cargar_datos_faiss(FAISS_DIR)
    RESUMEN_STATS = generar_resumen_estadistico(INDEX_DATA)
    logger.info("Cargados %d chunks de %d documentos.", len(INDEX_DATA), RESUMEN_STATS.get("total_documentos", 0))


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="es" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Visualizador de Metadatos FAISS — Kernel Boyacense</title>
  <!-- Google Fonts & Tailwind CSS -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            brand: {
              50: '#eef2ff',
              100: '#e0e7ff',
              500: '#6366f1',
              600: '#4f46e5',
              700: '#4338ca',
              800: '#3730a3',
              900: '#312e81',
              950: '#1e1b4b',
            },
            darkbg: '#0f172a',
            darkcard: '#1e293b',
            darkborder: '#334155'
          },
          fontFamily: {
            sans: ['Inter', 'sans-serif'],
            mono: ['JetBrains Mono', 'monospace']
          }
        }
      }
    }
  </script>
  <style>
    /* Custom Scrollbars */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #475569; }
    .glass {
      background: rgba(30, 41, 59, 0.7);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.08);
    }
  </style>
</head>
<body class="bg-darkbg text-slate-100 font-sans min-h-screen flex flex-col">

  <!-- Header / Navigation -->
  <header class="glass sticky top-0 z-50 px-6 py-4 flex items-center justify-between border-b border-slate-800">
    <div class="flex items-center gap-3">
      <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center shadow-lg shadow-indigo-500/30">
        <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path></svg>
      </div>
      <div>
        <h1 class="text-xl font-bold tracking-tight bg-gradient-to-r from-white via-slate-200 to-indigo-300 bg-clip-text text-transparent">
          Visualizador de Metadatos & Chunks FAISS
        </h1>
        <p class="text-xs text-slate-400">Kernel Boyacense — Visualización y Auditoría de Datos</p>
      </div>
    </div>

    <!-- Navigation Tabs -->
    <nav class="flex gap-2 bg-slate-900/60 p-1.5 rounded-xl border border-slate-800">
      <button onclick="switchTab('dashboard')" id="tab-dashboard" class="tab-btn px-4 py-2 text-xs font-semibold rounded-lg transition-all bg-indigo-600 text-white shadow-md">
        📊 Dashboard & Métricas
      </button>
      <button onclick="switchTab('explorer')" id="tab-explorer" class="tab-btn px-4 py-2 text-xs font-semibold rounded-lg transition-all text-slate-400 hover:text-white">
        📁 Explorador de Chunks & Metadatos
      </button>
    </nav>
  </header>

  <!-- Main Content Container -->
  <main class="flex-1 max-w-7xl w-full mx-auto p-6 space-y-6">

    <!-- 📊 TAB 1: DASHBOARD -->
    <section id="view-dashboard" class="space-y-6">
      <!-- Metric Cards -->
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div class="glass p-5 rounded-2xl border border-slate-800/80 hover:border-indigo-500/40 transition-all">
          <p class="text-xs font-medium text-slate-400 uppercase tracking-wider">Total Chunks Indexados</p>
          <div class="flex items-baseline gap-2 mt-2">
            <span id="stat-total-chunks" class="text-3xl font-extrabold text-white">--</span>
            <span class="text-xs text-emerald-400 font-medium">FAISS Index</span>
          </div>
        </div>

        <div class="glass p-5 rounded-2xl border border-slate-800/80 hover:border-indigo-500/40 transition-all">
          <p class="text-xs font-medium text-slate-400 uppercase tracking-wider">Documentos Únicos</p>
          <div class="flex items-baseline gap-2 mt-2">
            <span id="stat-total-docs" class="text-3xl font-extrabold text-indigo-400">--</span>
            <span class="text-xs text-slate-400 font-medium">Corpus ADL</span>
          </div>
        </div>

        <div class="glass p-5 rounded-2xl border border-slate-800/80 hover:border-indigo-500/40 transition-all">
          <p class="text-xs font-medium text-slate-400 uppercase tracking-wider">Tokens Totales</p>
          <div class="flex items-baseline gap-2 mt-2">
            <span id="stat-total-tokens" class="text-3xl font-extrabold text-violet-400">--</span>
            <span class="text-xs text-slate-400 font-medium">tokens</span>
          </div>
        </div>

        <div class="glass p-5 rounded-2xl border border-slate-800/80 hover:border-indigo-500/40 transition-all">
          <p class="text-xs font-medium text-slate-400 uppercase tracking-wider">Promedio Tokens/Chunk</p>
          <div class="flex items-baseline gap-2 mt-2">
            <span id="stat-avg-tokens" class="text-3xl font-extrabold text-amber-400">--</span>
            <span class="text-xs text-slate-400 font-medium">t/chunk</span>
          </div>
        </div>
      </div>

      <!-- Charts Section -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div class="glass p-6 rounded-2xl border border-slate-800">
          <h3 class="text-sm font-semibold text-slate-200 mb-4 flex items-center gap-2">
            <span class="w-2 h-2 rounded-full bg-indigo-500"></span>
            Distribución por Formato de Archivo
          </h3>
          <div class="h-64 relative flex items-center justify-center">
            <canvas id="chart-formats"></canvas>
          </div>
        </div>

        <div class="glass p-6 rounded-2xl border border-slate-800">
          <h3 class="text-sm font-semibold text-slate-200 mb-4 flex items-center gap-2">
            <span class="w-2 h-2 rounded-full bg-violet-500"></span>
            Top 10 Documentos con Mayor Cantidad de Chunks
          </h3>
          <div class="h-64 relative">
            <canvas id="chart-top-docs"></canvas>
          </div>
        </div>
      </div>

      <!-- Export Actions Card -->
      <div class="glass p-6 rounded-2xl border border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div>
          <h4 class="font-semibold text-slate-200">Exportar Base de Datos & Metadatos Parseados</h4>
          <p class="text-xs text-slate-400">Descarga la totalidad de fragmentos y metadatos estructurados en tu formato preferido.</p>
        </div>
        <div class="flex items-center gap-3">
          <a href="/api/export?format=json" download="faiss_chunks.json" class="px-4 py-2 text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-indigo-300 border border-slate-700 rounded-xl transition-all flex items-center gap-2">
            ⬇ JSON Completo
          </a>
          <a href="/api/export?format=csv" download="faiss_chunks.csv" class="px-4 py-2 text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-emerald-300 border border-slate-700 rounded-xl transition-all flex items-center gap-2">
            ⬇ CSV Tabular
          </a>
        </div>
      </div>
    </section>

    <!-- 📁 TAB 2: EXPLORADOR DE CHUNKS & METADATOS -->
    <section id="view-explorer" class="space-y-6 hidden">
      <!-- Search & Filters Bar -->
      <div class="glass p-4 rounded-2xl border border-slate-800 flex flex-wrap items-center justify-between gap-4">
        <div class="flex-1 min-w-[280px] relative">
          <svg class="w-4 h-4 text-slate-400 absolute left-3.5 top-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
          <input type="text" id="filter-search" oninput="debounceFetchChunks()" placeholder="Filtrar por texto, doc_id, chunk_id o fuente..." class="w-full bg-slate-900/80 border border-slate-700 text-sm text-slate-200 pl-10 pr-4 py-2.5 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:outline-none">
        </div>

        <div class="flex flex-wrap items-center gap-3">
          <!-- Formato Filter -->
          <select id="filter-formato" onchange="fetchChunks()" class="bg-slate-900 border border-slate-700 text-xs text-slate-300 px-3 py-2.5 rounded-xl focus:outline-none">
            <option value="">Todos los Formatos</option>
          </select>

          <!-- Fenomeno Filter -->
          <select id="filter-fenomeno" onchange="fetchChunks()" class="bg-slate-900 border border-slate-700 text-xs text-slate-300 px-3 py-2.5 rounded-xl focus:outline-none">
            <option value="">Todos los Fenómenos</option>
          </select>

          <!-- Documento Filter -->
          <select id="filter-doc" onchange="fetchChunks()" class="bg-slate-900 border border-slate-700 text-xs text-slate-300 px-3 py-2.5 rounded-xl focus:outline-none max-w-[200px]">
            <option value="">Todos los Documentos</option>
          </select>
        </div>
      </div>

      <!-- Chunks Table displaying all metadata fields -->
      <div class="glass rounded-2xl border border-slate-800 overflow-hidden">
        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs text-slate-300">
            <thead class="bg-slate-900/90 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
              <tr>
                <th class="py-3 px-3">FAISS ID</th>
                <th class="py-3 px-3">chunk_id</th>
                <th class="py-3 px-3">doc_id</th>
                <th class="py-3 px-3">posicion</th>
                <th class="py-3 px-3">formato</th>
                <th class="py-3 px-3">fenomeno</th>
                <th class="py-3 px-3">num_tokens</th>
                <th class="py-3 px-3">fuente</th>
                <th class="py-3 px-3">texto (vista previa)</th>
                <th class="py-3 px-3 text-right">Metadatos</th>
              </tr>
            </thead>
            <tbody id="chunks-table-body" class="divide-y divide-slate-800/60 font-mono">
              <!-- Rendered via JS -->
            </tbody>
          </table>
        </div>

        <!-- Pagination Controls -->
        <div class="p-4 bg-slate-900/40 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400 font-sans">
          <div>
            Mostrando <span id="pag-start" class="font-semibold text-white">0</span> a <span id="pag-end" class="font-semibold text-white">0</span> de <span id="pag-total" class="font-semibold text-white">0</span> chunks
          </div>
          <div class="flex items-center gap-2">
            <button onclick="changePage(-1)" id="btn-prev" class="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-200 rounded-lg">Anterior</button>
            <span id="pag-page" class="font-semibold text-indigo-400 px-2">Página 1</span>
            <button onclick="changePage(1)" id="btn-next" class="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-200 rounded-lg">Siguiente</button>
          </div>
        </div>
      </div>
    </section>

  </main>

  <!-- 🔍 MODAL: VISOR COMPLETO DE CHUNK Y METADATOS -->
  <div id="chunk-modal" class="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm hidden items-center justify-center p-4">
    <div class="glass max-w-4xl w-full max-h-[90vh] flex flex-col rounded-2xl border border-slate-700 shadow-2xl overflow-hidden">
      <!-- Modal Header -->
      <div class="p-5 border-b border-slate-800 flex items-center justify-between bg-slate-900/80">
        <div>
          <div class="flex items-center gap-3">
            <span id="modal-chunk-id" class="text-base font-bold text-indigo-300 font-mono">DOC-0001-chunk-000</span>
            <span id="modal-formato-badge" class="px-2 py-0.5 text-[10px] font-semibold bg-slate-800 text-slate-300 rounded border border-slate-700 uppercase">PDF</span>
            <span id="modal-faiss-id-badge" class="px-2 py-0.5 text-[10px] font-mono bg-indigo-950 text-indigo-300 rounded border border-indigo-800">FAISS ID: --</span>
          </div>
          <p id="modal-doc-id-header" class="text-xs text-slate-400 mt-1">Documento: ---</p>
        </div>
        <button onclick="closeModal()" class="text-slate-400 hover:text-white p-2">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
        </button>
      </div>

      <!-- Modal Body -->
      <div class="p-6 overflow-y-auto space-y-6 flex-1">
        
        <!-- Structured Metadata Grid -->
        <div>
          <h4 class="text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-3 flex items-center gap-2">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
            Tabla de Metadatos del Fragmento
          </h4>
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
            <div class="bg-slate-900/90 p-3 rounded-xl border border-slate-800">
              <span class="text-[10px] uppercase font-semibold text-slate-400 block">doc_id</span>
              <span id="meta-doc-id" class="font-mono text-slate-200 font-medium break-all">--</span>
            </div>
            <div class="bg-slate-900/90 p-3 rounded-xl border border-slate-800">
              <span class="text-[10px] uppercase font-semibold text-slate-400 block">chunk_id</span>
              <span id="meta-chunk-id" class="font-mono text-indigo-300 font-medium break-all">--</span>
            </div>
            <div class="bg-slate-900/90 p-3 rounded-xl border border-slate-800">
              <span class="text-[10px] uppercase font-semibold text-slate-400 block">formato</span>
              <span id="meta-formato" class="font-mono text-emerald-400 font-medium">--</span>
            </div>
            <div class="bg-slate-900/90 p-3 rounded-xl border border-slate-800">
              <span class="text-[10px] uppercase font-semibold text-slate-400 block">posicion</span>
              <span id="meta-posicion" class="font-mono text-amber-400 font-medium">--</span>
            </div>
            <div class="bg-slate-900/90 p-3 rounded-xl border border-slate-800">
              <span class="text-[10px] uppercase font-semibold text-slate-400 block">num_tokens</span>
              <span id="meta-tokens" class="font-mono text-violet-400 font-medium">--</span>
            </div>
            <div class="bg-slate-900/90 p-3 rounded-xl border border-slate-800">
              <span class="text-[10px] uppercase font-semibold text-slate-400 block">fenomeno</span>
              <span id="meta-fenomeno" class="font-mono text-sky-400 font-medium">--</span>
            </div>
            <div class="bg-slate-900/90 p-3 rounded-xl border border-slate-800 md:col-span-2">
              <span class="text-[10px] uppercase font-semibold text-slate-400 block">fuente (ruta de archivo)</span>
              <span id="meta-fuente" class="font-mono text-slate-300 text-[11px] break-all">--</span>
            </div>
          </div>
        </div>

        <!-- Text Content Box -->
        <div>
          <h4 class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-2">
            <svg class="w-4 h-4 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h7"></path></svg>
            Contenido de Texto (campo: texto)
          </h4>
          <div id="modal-text-content" class="p-4 bg-slate-950 rounded-xl border border-slate-800 text-slate-200 font-mono text-xs leading-relaxed whitespace-pre-wrap select-all max-h-72 overflow-y-auto">
            Cargando contenido...
          </div>
        </div>

        <!-- Raw Metadata JSON Viewer -->
        <div>
          <h4 class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Metadatos Internos Extra (Diccionario JSON)</h4>
          <pre id="modal-metadata-json" class="p-4 bg-slate-950 rounded-xl border border-slate-800 text-emerald-400 font-mono text-[11px] overflow-x-auto max-h-48">
{}
          </pre>
        </div>
      </div>

      <!-- Modal Footer -->
      <div class="p-4 border-t border-slate-800 bg-slate-900/60 flex justify-end">
        <button onclick="closeModal()" class="px-5 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-xl shadow-lg transition-all">
          Cerrar Visor
        </button>
      </div>
    </div>
  </div>

  <!-- JavaScript Client Logic -->
  <script>
    let statsData = {};
    let currentChunks = [];
    let currentPage = 1;
    const pageSize = 20;
    let totalItems = 0;
    let debounceTimer;

    document.addEventListener('DOMContentLoaded', () => {
      loadStats();
      fetchChunks();
    });

    function switchTab(tabName) {
      document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('bg-indigo-600', 'text-white', 'shadow-md');
        btn.classList.add('text-slate-400');
      });
      document.getElementById(`tab-${tabName}`).classList.add('bg-indigo-600', 'text-white', 'shadow-md');

      document.querySelectorAll('main > section').forEach(sec => sec.classList.add('hidden'));
      document.getElementById(`view-${tabName}`).classList.remove('hidden');
    }

    async function loadStats() {
      try {
        const res = await fetch('/api/stats');
        statsData = await res.json();

        document.getElementById('stat-total-chunks').innerText = statsData.total_chunks ? statsData.total_chunks.toLocaleString() : '0';
        document.getElementById('stat-total-docs').innerText = statsData.total_documentos || 0;
        document.getElementById('stat-total-tokens').innerText = statsData.tokens_totales ? statsData.tokens_totales.toLocaleString() : '0';
        document.getElementById('stat-avg-tokens').innerText = statsData.promedio_tokens_por_chunk || 0;

        populateDropdowns();
        renderCharts();
      } catch (err) {
        console.error("Error loading stats:", err);
      }
    }

    function populateDropdowns() {
      const formatoSelect = document.getElementById('filter-formato');
      formatoSelect.innerHTML = '<option value="">Todos los Formatos</option>';
      Object.keys(statsData.desglose_formatos || {}).forEach(fmt => {
        formatoSelect.innerHTML += `<option value="${fmt}">${fmt} (${statsData.desglose_formatos[fmt]})</option>`;
      });

      const fenSelect = document.getElementById('filter-fenomeno');
      fenSelect.innerHTML = '<option value="">Todos los Fenómenos</option>';
      Object.keys(statsData.desglose_fenomenos || {}).forEach(fen => {
        fenSelect.innerHTML += `<option value="${fen}">${fen} (${statsData.desglose_fenomenos[fen]})</option>`;
      });

      const docSelect = document.getElementById('filter-doc');
      docSelect.innerHTML = '<option value="">Todos los Documentos</option>';
      Object.keys(statsData.chunks_por_documento || {}).forEach(doc => {
        docSelect.innerHTML += `<option value="${doc}">${doc}</option>`;
      });
    }

    function renderCharts() {
      // Formats Doughnut Chart
      const fmtCtx = document.getElementById('chart-formats').getContext('2d');
      new Chart(fmtCtx, {
        type: 'doughnut',
        data: {
          labels: Object.keys(statsData.desglose_formatos || {}),
          datasets: [{
            data: Object.values(statsData.desglose_formatos || {}),
            backgroundColor: ['#6366f1', '#8b5cf6', '#ec4899', '#10b981', '#f59e0b', '#3b82f6', '#64748b']
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { position: 'right', labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } } } }
        }
      });

      // Top Docs Bar Chart
      const topDocs = Object.entries(statsData.chunks_por_documento || {})
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10);

      const docCtx = document.getElementById('chart-top-docs').getContext('2d');
      new Chart(docCtx, {
        type: 'bar',
        data: {
          labels: topDocs.map(d => d[0].length > 20 ? d[0].substring(0, 17) + '...' : d[0]),
          datasets: [{
            label: 'Chunks',
            data: topDocs.map(d => d[1]),
            backgroundColor: '#6366f1',
            borderRadius: 6
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { display: false } },
            y: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } }
          }
        }
      });
    }

    function debounceFetchChunks() {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => { currentPage = 1; fetchChunks(); }, 300);
    }

    async function fetchChunks() {
      const q = document.getElementById('filter-search').value;
      const formato = document.getElementById('filter-formato').value;
      const fenomeno = document.getElementById('filter-fenomeno').value;
      const doc = document.getElementById('filter-doc').value;

      const url = `/api/chunks?page=${currentPage}&page_size=${pageSize}&q=${encodeURIComponent(q)}&formato=${encodeURIComponent(formato)}&fenomeno=${encodeURIComponent(fenomeno)}&doc=${encodeURIComponent(doc)}`;

      try {
        const res = await fetch(url);
        const data = await res.json();
        currentChunks = data.items;
        totalItems = data.total;

        renderChunksTable();
        updatePagination();
      } catch (err) {
        console.error("Error fetching chunks:", err);
      }
    }

    function renderChunksTable() {
      const tbody = document.getElementById('chunks-table-body');
      tbody.innerHTML = '';

      if (!currentChunks || currentChunks.length === 0) {
        tbody.innerHTML = `<tr><td colspan="10" class="py-8 text-center font-sans text-slate-500">No se encontraron fragmentos con los filtros seleccionados.</td></tr>`;
        return;
      }

      currentChunks.forEach(item => {
        const preview = item.texto ? (item.texto.length > 80 ? item.texto.substring(0, 80) + '...' : item.texto) : '';
        const fuenteAbbrev = item.fuente ? (item.fuente.length > 25 ? '...' + item.fuente.substring(item.fuente.length - 25) : item.fuente) : '';
        const fenomenoVal = item.fenomeno !== null && item.fenomeno !== undefined ? item.fenomeno : 'Sin asignar';

        const row = document.createElement('tr');
        row.className = 'hover:bg-slate-800/40 transition-colors border-b border-slate-800/40 text-[11px]';
        row.innerHTML = `
          <td class="py-2.5 px-3 text-slate-400">${item.faiss_id}</td>
          <td class="py-2.5 px-3 font-semibold text-indigo-300">${escapeHtml(item.chunk_id)}</td>
          <td class="py-2.5 px-3 text-slate-200 font-sans font-medium max-w-[120px] truncate" title="${escapeHtml(item.doc_id)}">${escapeHtml(item.doc_id)}</td>
          <td class="py-2.5 px-3 text-amber-400">#${item.posicion}</td>
          <td class="py-2.5 px-3 font-sans"><span class="px-1.5 py-0.5 text-[10px] font-semibold bg-slate-800 text-emerald-400 rounded border border-slate-700">${escapeHtml(item.formato)}</span></td>
          <td class="py-2.5 px-3 font-sans text-sky-300 text-[10px]">${escapeHtml(fenomenoVal)}</td>
          <td class="py-2.5 px-3 text-violet-400 font-bold">${item.num_tokens}</td>
          <td class="py-2.5 px-3 text-slate-400 text-[10px] max-w-[140px] truncate" title="${escapeHtml(item.fuente)}">${escapeHtml(fuenteAbbrev)}</td>
          <td class="py-2.5 px-3 text-slate-300 font-sans leading-snug max-w-[200px] truncate" title="${escapeHtml(item.texto)}">${escapeHtml(preview)}</td>
          <td class="py-2.5 px-3 text-right font-sans">
            <button onclick="openModal(${item.faiss_id})" class="px-2 py-1 text-[10px] font-semibold bg-indigo-600/20 hover:bg-indigo-600 text-indigo-300 hover:text-white rounded-lg transition-all border border-indigo-500/30 whitespace-nowrap">
              🔍 Ver Metadatos
            </button>
          </td>
        `;
        tbody.appendChild(row);
      });
    }

    function updatePagination() {
      const start = totalItems === 0 ? 0 : (currentPage - 1) * pageSize + 1;
      const end = Math.min(currentPage * pageSize, totalItems);

      document.getElementById('pag-start').innerText = start;
      document.getElementById('pag-end').innerText = end;
      document.getElementById('pag-total').innerText = totalItems;
      document.getElementById('pag-page').innerText = `Página ${currentPage}`;

      document.getElementById('btn-prev').disabled = currentPage <= 1;
      document.getElementById('btn-next').disabled = end >= totalItems;
    }

    function changePage(delta) {
      currentPage += delta;
      fetchChunks();
    }

    function openModal(faissId) {
      const item = currentChunks.find(c => c.faiss_id === faissId);
      if (!item) return;

      document.getElementById('modal-chunk-id').innerText = item.chunk_id;
      document.getElementById('modal-formato-badge').innerText = (item.formato || '').toUpperCase();
      document.getElementById('modal-faiss-id-badge').innerText = `FAISS ID: ${item.faiss_id}`;
      document.getElementById('modal-doc-id-header').innerText = `Documento: ${item.doc_id}`;

      // Populate structured 8 fields
      document.getElementById('meta-doc-id').innerText = item.doc_id;
      document.getElementById('meta-chunk-id').innerText = item.chunk_id;
      document.getElementById('meta-formato').innerText = item.formato;
      document.getElementById('meta-posicion').innerText = item.posicion;
      document.getElementById('meta-tokens').innerText = item.num_tokens;
      document.getElementById('meta-fenomeno').innerText = item.fenomeno !== null && item.fenomeno !== undefined ? item.fenomeno : 'Sin asignar';
      document.getElementById('meta-fuente').innerText = item.fuente || 'Sin fuente';

      document.getElementById('modal-text-content').innerText = item.texto;
      document.getElementById('modal-metadata-json').innerText = JSON.stringify(item.metadata || {}, null, 2);

      document.getElementById('chunk-modal').classList.remove('hidden');
      document.getElementById('chunk-modal').classList.add('flex');
    }

    function closeModal() {
      document.getElementById('chunk-modal').classList.add('hidden');
      document.getElementById('chunk-modal').classList.remove('flex');
    }

    function escapeHtml(str) {
      if (!str) return '';
      return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
    }
  </script>
</body>
</html>
"""


class RequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, status: int = 200):
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query_params = urllib.parse.parse_qs(parsed.query)

        if path in ["/", "/index.html"]:
            self._send_html(HTML_TEMPLATE)
            return

        if path == "/api/stats":
            self._send_json(RESUMEN_STATS)
            return

        if path == "/api/chunks":
            page = int(query_params.get("page", [1])[0])
            page_size = int(query_params.get("page_size", [20])[0])
            q = query_params.get("q", [""])[0].lower().strip()
            formato = query_params.get("formato", [""])[0].strip()
            fenomeno = query_params.get("fenomeno", [""])[0].strip()
            doc = query_params.get("doc", [""])[0].strip()

            filtered = INDEX_DATA
            if formato:
                filtered = [item for item in filtered if item["formato"] == formato]
            if fenomeno:
                if fenomeno == "Sin asignar":
                    filtered = [item for item in filtered if item["fenomeno"] is None or item["fenomeno"] == "Sin asignar"]
                else:
                    filtered = [item for item in filtered if str(item["fenomeno"]) == fenomeno]
            if doc:
                filtered = [item for item in filtered if item["doc_id"] == doc]
            if q:
                filtered = [
                    item for item in filtered
                    if q in item["texto"].lower() or q in item["doc_id"].lower() or q in item["chunk_id"].lower() or q in item["fuente"].lower()
                ]

            total = len(filtered)
            start = (page - 1) * page_size
            end = start + page_size
            items = filtered[start:end]

            self._send_json({"total": total, "page": page, "page_size": page_size, "items": items})
            return

        if path == "/api/export":
            fmt = query_params.get("format", ["json"])[0]
            if fmt == "csv":
                import csv
                from io import StringIO
                output = StringIO()
                writer = csv.DictWriter(output, fieldnames=["faiss_id", "doc_id", "chunk_id", "formato", "fenomeno", "posicion", "num_tokens", "fuente", "texto"])
                writer.writeheader()
                for item in INDEX_DATA:
                    writer.writerow({k: item.get(k, "") for k in writer.fieldnames})
                body = output.getvalue().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", "attachment; filename=faiss_chunks.csv")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            else:
                self._send_json(INDEX_DATA)
                return

        self._send_json({"error": "Ruta no encontrada"}, status=404)


def run_server(port: int = PORT):
    cargar_indice_y_metadatos()
    server_address = ("", port)
    httpd = HTTPServer(server_address, RequestHandler)
    logger.info("Servidor de Visualización de Metadatos FAISS iniciado en http://localhost:%d", port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Deteniendo el servidor web...")
        httpd.server_close()


if __name__ == "__main__":
    port_arg = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    run_server(port_arg)
