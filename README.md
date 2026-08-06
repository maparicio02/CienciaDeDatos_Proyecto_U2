# Visualizador Acústico Espaciotemporal 3D

Aplicación desarrollada en Python para el análisis y visualización interactiva de archivos de audio (WAV y MP3). El sistema extrae características acústicas, reduce su dimensionalidad mediante Análisis de Componentes Principales (PCA) y representa la evolución temporal del audio como una trayectoria tridimensional sincronizada con la reproducción.

## Características
- Carga y procesamiento de archivos WAV y MP3.
- Extracción de características acústicas (MFCC, Chroma, RMS, Centroide, Rolloff, ZCR, Flatness, entre otras).
- Reducción de dimensionalidad mediante PCA a tres componentes principales.
- Visualización 3D en tiempo real sincronizada con la reproducción del audio.
- Dashboard con forma de onda, métricas y controles interactivos.
- Exportación de la visualización en PNG y de los datos procesados en CSV.

## Tecnologías utilizadas
Python
Librosa
NumPy
Scikit-learn
Matplotlib
Pygame

## Objetivo
Proporcionar una herramienta que facilite la exploración y comprensión de la estructura acústica de señales de audio mediante técnicas de procesamiento digital de señales, extracción de características y visualización tridimensional interactiva.
