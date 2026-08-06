
from __future__ import annotations

from pathlib import Path
from queue import Empty, Full, Queue
import os
import sys
import threading
import warnings

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pygame
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

CARPETA_PROYECTO = Path(__file__).resolve().parent

# El programa busca primero sonido.wav y luego sonido.mp3. Si no encuentra
# ninguno, toma el primer archivo WAV o MP3 disponible en la carpeta.
EXTENSIONES_AUDIO_ADMITIDAS = (".wav", ".mp3")
NOMBRES_PRIORITARIOS = ("sonido.wav", "sonido.mp3")


def localizar_audio() -> Path | None:
    # Permite abrir un audio específico pasando su ruta como argumento:
    # python Visualizador_Acustico_3D_PRO.py "mi_audio.mp3"
    if len(sys.argv) > 1:
        ruta_argumento = Path(sys.argv[1]).expanduser()
        if (
            ruta_argumento.is_file()
            and ruta_argumento.suffix.lower() in EXTENSIONES_AUDIO_ADMITIDAS
        ):
            return ruta_argumento.resolve()

    for nombre in NOMBRES_PRIORITARIOS:
        candidato = CARPETA_PROYECTO / nombre
        if candidato.exists():
            return candidato

    candidatos = sorted(
        archivo
        for archivo in CARPETA_PROYECTO.iterdir()
        if archivo.is_file()
        and archivo.suffix.lower() in EXTENSIONES_AUDIO_ADMITIDAS
    )
    return candidatos[0] if candidatos else None


RUTA_AUDIO = localizar_audio()
RUTA_CAPTURA = CARPETA_PROYECTO / "captura_manifold_acustico.png"
RUTA_CSV = CARPETA_PROYECTO / "datos_manifold_acustico.csv"

FRECUENCIA_OBJETIVO = 22050
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
VENTANA_ANALISIS = 4096
PASO_ANALISIS_SEGUNDOS = 0.18
INTERVALO_INTERFAZ_MS = 40

# Control de densidad visual.
MAXIMO_NODOS_MEMORIA = 3500
MAXIMO_NODOS_VISIBLES = 1100
MAXIMO_SEGMENTOS_VISIBLES = 850

# Suavizado exponencial del recorrido.
# Un valor mayor conserva más movimiento real y evita una trayectoria plana.
FACTOR_SUAVIZADO = 0.50

# Expansión visual del manifold. No altera el audio ni el PCA; únicamente
# redistribuye sus tres componentes para aprovechar mejor el espacio 3D.
# CP2 y CP3 se amplían más porque suelen quedar comprimidas frente a CP1.
# Expansión fuerte para abrir la nube y hacer visibles los recorridos.
# Los límites del gráfico se calculan con percentiles robustos, por eso el
# manifold llega casi hasta los bordes sin que unos pocos valores extremos
# dejen toda la nube comprimida en el centro.
EXPANSION_GLOBAL = 3.45
EXPANSION_POR_EJE = np.array([1.55, 2.65, 3.25], dtype=float)
POTENCIA_SEPARACION = 0.66
PERCENTIL_ESCALA_PCA = 87.0
PERCENTIL_LIMITE_INFERIOR = 1.0
PERCENTIL_LIMITE_SUPERIOR = 99.0
MARGEN_LIMITE_3D = 0.025

# Zoom con rueda del mouse.
FACTOR_ZOOM_ENTRADA = 0.84
FACTOR_ZOOM_SALIDA = 1.19
ZOOM_MINIMO_RELATIVO = 0.12
ZOOM_MAXIMO_RELATIVO = 5.00

TAMANIO_NODO_MIN = 12.0
TAMANIO_NODO_MAX = 155.0

# Paleta cronológica solicitada:
# inicio amarillo -> azul -> final morado.
MAPA_TIEMPO = LinearSegmentedColormap.from_list(
    "tiempo_audio",
    [
        "#fde725",  # amarillo
        "#77e6c4",  # verde agua
        "#22b8cf",  # celeste
        "#2563eb",  # azul
        "#6d28d9",  # violeta
        "#3b0764",  # morado oscuro
    ],
)

COLOR_FONDO = "#050816"
COLOR_FONDO_2 = "#081126"
COLOR_PANEL = "#0b1328"
COLOR_PANEL_2 = "#111d38"
COLOR_PANEL_3 = "#162441"
COLOR_BORDE = "#28436f"
COLOR_TEXTO = "#f8fafc"
COLOR_SECUNDARIO = "#9fb0ca"
COLOR_ACENTO = "#2dd4bf"
COLOR_ACENTO_2 = "#8b5cf6"
COLOR_ACENTO_3 = "#f59e0b"
COLOR_PELIGRO = "#ef4444"
COLOR_CURSOR = "#ffffff"
COLOR_EXITO = "#34d399"
COLOR_INFO = "#38bdf8"
COLOR_MAGENTA = "#f472b6"

VOLUMEN_INICIAL = 0.82
VELOCIDAD_AUTOGIRO = 0.22

MAPA_ACTIVIDAD = LinearSegmentedColormap.from_list(
    "actividad_audio",
    ["#2dd4bf", "#38bdf8", "#8b5cf6", "#f59e0b", "#fb7185"],
)


# ============================================================
# 2. UTILIDADES
# ============================================================


def terminar_con_error(mensaje: str) -> None:
    print("\n" + "=" * 76)
    print("ERROR")
    print("=" * 76)
    print(mensaje)
    sys.exit(1)


def limpiar_cola(cola: Queue) -> None:
    while True:
        try:
            cola.get_nowait()
        except Empty:
            break


def limitar_eje(valores: np.ndarray) -> tuple[float, float]:
    """
    Calcula límites robustos para que el recorrido use casi todo el gráfico.

    Se ignora únicamente el 1 % más extremo de cada lado. Así, los valores
    atípicos no comprimen la nube central y los nodos llegan casi al borde.
    """
    valores = np.asarray(valores, dtype=float)
    valores = valores[np.isfinite(valores)]

    if valores.size == 0:
        return -1.0, 1.0

    minimo, maximo = np.percentile(
        valores,
        [PERCENTIL_LIMITE_INFERIOR, PERCENTIL_LIMITE_SUPERIOR],
    )
    minimo = float(minimo)
    maximo = float(maximo)
    rango = maximo - minimo

    if rango < 1e-8:
        centro = (minimo + maximo) / 2.0
        return centro - 1.0, centro + 1.0

    margen = rango * MARGEN_LIMITE_3D
    return minimo - margen, maximo + margen


def formatear_tiempo(segundos: float) -> str:
    segundos = max(0.0, float(segundos))
    minutos = int(segundos // 60)
    resto = segundos - minutos * 60
    return f"{minutos:02d}:{resto:05.2f}"


def suavizar_serie(serie: np.ndarray, ventana: int = 7) -> np.ndarray:
    """Suavizado simple usado solo para calcular límites estables del PCA."""
    serie = np.asarray(serie, dtype=float)
    if len(serie) < 3:
        return serie.copy()

    ventana = max(3, min(int(ventana), len(serie)))
    if ventana % 2 == 0:
        ventana -= 1

    kernel = np.ones(ventana, dtype=float) / ventana
    resultado = np.empty_like(serie)

    for columna in range(serie.shape[1]):
        resultado[:, columna] = np.convolve(
            serie[:, columna], kernel, mode="same"
        )

    return resultado


def percentiles_seguros(vector: np.ndarray) -> tuple[float, float]:
    inferior, superior = np.percentile(np.asarray(vector, dtype=float), [5, 95])
    inferior = float(inferior)
    superior = float(superior)

    if superior - inferior < 1e-12:
        superior = inferior + 1.0

    return inferior, superior


def normalizar_por_rango(valor: float, inferior: float, superior: float) -> float:
    return float(np.clip((valor - inferior) / (superior - inferior), 0.0, 1.0))


def expandir_coordenadas_pca(puntos: np.ndarray) -> np.ndarray:
    """
    Convierte el PCA original a un espacio visual más abierto.

    1. Centra cada componente con la mediana de calibración.
    2. Divide cada eje por una escala robusta.
    3. Usa una potencia menor que uno para separar puntos cercanos al centro.
    4. Amplifica CP2 y CP3 para que el recorrido ocupe realmente el volumen 3D.

    La transformación es determinista y se aplica igual a la calibración y a
    cada ventana analizada durante la reproducción.
    """
    arreglo = np.asarray(puntos, dtype=float)
    era_vector = arreglo.ndim == 1
    if era_vector:
        arreglo = arreglo.reshape(1, -1)

    normalizado = (arreglo - CENTRO_PCA) / ESCALA_PCA
    separado = np.sign(normalizado) * np.power(
        np.abs(normalizado), POTENCIA_SEPARACION
    )
    expandido = separado * EXPANSION_GLOBAL * EXPANSION_POR_EJE

    return expandido[0] if era_vector else expandido


# ============================================================
# 3. CARGA DEL AUDIO
# ============================================================

if RUTA_AUDIO is None:
    terminar_con_error(
        "No se encontró ningún audio WAV o MP3.\n"
        "Coloca sonido.wav o sonido.mp3 junto al programa.\n"
        f"Carpeta revisada: {CARPETA_PROYECTO}"
    )

print("=" * 76)
print("VISUALIZADOR ACÚSTICO ESPACIOTEMPORAL 3D")
print("=" * 76)
print(f"Archivo: {RUTA_AUDIO.name}")

try:
    audio, frecuencia_muestreo = librosa.load(
        RUTA_AUDIO,
        sr=FRECUENCIA_OBJETIVO,
        mono=True,
    )
except Exception as error:
    terminar_con_error(
        "No se pudo cargar el audio WAV/MP3.\n"
        f"Archivo: {RUTA_AUDIO.name}\n"
        f"Detalle: {error}\n"
        "Para MP3, instala o actualiza soundfile y librosa."
    )

if audio.size == 0:
    terminar_con_error("El archivo de audio está vacío.")

audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(
    np.float32,
    copy=False,
)

duracion = float(librosa.get_duration(y=audio, sr=frecuencia_muestreo))

if duracion < 0.5:
    terminar_con_error("El audio debe durar al menos 0.5 segundos.")

if float(np.max(np.abs(audio))) < 1e-8:
    terminar_con_error("El archivo parece contener únicamente silencio.")

print(f"Duración: {duracion:.2f} segundos")
print(f"Frecuencia de muestreo: {frecuencia_muestreo} Hz")
print(f"Muestras: {len(audio):,}")


# ============================================================
# 4. CALIBRACIÓN DEL ESPACIO PCA
# ============================================================
# Se calibra una sola vez antes de mostrar la interfaz. Esto mantiene
# estables los ejes 3D. Los nodos que se muestran se calculan después,
# durante la reproducción, ventana por ventana.

print("\nCalibrando PCA con 40 MFCC y variables espectrales...")

try:
    mfcc_cal = librosa.feature.mfcc(
        y=audio,
        sr=frecuencia_muestreo,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    chroma_cal = librosa.feature.chroma_stft(
        y=audio,
        sr=frecuencia_muestreo,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    rms_cal = librosa.feature.rms(
        y=audio,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
    )
    centroide_cal = librosa.feature.spectral_centroid(
        y=audio,
        sr=frecuencia_muestreo,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    ancho_cal = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=frecuencia_muestreo,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    rolloff_cal = librosa.feature.spectral_rolloff(
        y=audio,
        sr=frecuencia_muestreo,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        roll_percent=0.85,
    )
    zcr_cal = librosa.feature.zero_crossing_rate(
        y=audio,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
    )
    flatness_cal = librosa.feature.spectral_flatness(
        y=audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    flujo_cal = librosa.onset.onset_strength(
        y=audio,
        sr=frecuencia_muestreo,
        hop_length=HOP_LENGTH,
    )
except Exception as error:
    terminar_con_error(f"No se pudieron extraer las características:\n{error}")

matrices_calibracion = [
    mfcc_cal,
    chroma_cal,
    rms_cal,
    centroide_cal,
    ancho_cal,
    rolloff_cal,
    zcr_cal,
    flatness_cal,
]

numero_frames = min(matriz.shape[1] for matriz in matrices_calibracion)
if numero_frames < 3:
    terminar_con_error("No hay suficientes frames para aplicar PCA.")

matrices_calibracion = [
    matriz[:, :numero_frames] for matriz in matrices_calibracion
]
X_calibracion = np.vstack(matrices_calibracion).T
X_calibracion = np.nan_to_num(
    X_calibracion,
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)

MAXIMO_FRAMES_CALIBRACION = 18000
if len(X_calibracion) > MAXIMO_FRAMES_CALIBRACION:
    indices_ajuste = np.linspace(
        0,
        len(X_calibracion) - 1,
        MAXIMO_FRAMES_CALIBRACION,
        dtype=int,
    )
    X_ajuste = X_calibracion[indices_ajuste]
else:
    X_ajuste = X_calibracion

escalador = StandardScaler()
X_ajuste_escalado = escalador.fit_transform(X_ajuste)

pca = PCA(n_components=3, random_state=0)
coordenadas_calibracion_pca = pca.fit_transform(X_ajuste_escalado)

# Parámetros robustos utilizados por expandir_coordenadas_pca().
CENTRO_PCA = np.median(coordenadas_calibracion_pca, axis=0)
ESCALA_PCA = np.percentile(
    np.abs(coordenadas_calibracion_pca - CENTRO_PCA),
    PERCENTIL_ESCALA_PCA,
    axis=0,
)
ESCALA_PCA = np.where(ESCALA_PCA < 1e-9, 1.0, ESCALA_PCA)

coordenadas_calibracion = expandir_coordenadas_pca(
    coordenadas_calibracion_pca
)
coordenadas_calibracion_suaves = suavizar_serie(
    coordenadas_calibracion, 5
)

varianza_individual = pca.explained_variance_ratio_ * 100.0
varianza_total = float(np.sum(varianza_individual))

rms_inf, rms_sup = percentiles_seguros(rms_cal[0])
centroide_inf, centroide_sup = percentiles_seguros(centroide_cal[0])
zcr_inf, zcr_sup = percentiles_seguros(zcr_cal[0])
flujo_inf, flujo_sup = percentiles_seguros(flujo_cal)

print(f"Frames utilizados para calibración: {len(X_ajuste):,}")
print(f"Variables acústicas por frame: {X_ajuste.shape[1]}")
print("Varianza explicada por PCA:")
print(f"CP1: {varianza_individual[0]:.2f}%")
print(f"CP2: {varianza_individual[1]:.2f}%")
print(f"CP3: {varianza_individual[2]:.2f}%")
print(f"Total: {varianza_total:.2f}%")

# Liberar matrices grandes.
del X_calibracion, X_ajuste_escalado, matrices_calibracion


# ============================================================
# 5. REPRODUCTOR DE AUDIO
# ============================================================

try:
    pygame.mixer.init()
    pygame.mixer.music.load(str(RUTA_AUDIO))
    pygame.mixer.music.set_volume(VOLUMEN_INICIAL)
except Exception as error:
    terminar_con_error(
        "No se pudo inicializar pygame.mixer.\n"
        f"Detalle: {error}\n"
        "Verifica el dispositivo de sonido y el formato del archivo."
    )


# ============================================================
# 6. ESTADO, COLAS Y DATOS
# ============================================================

cola_solicitudes: Queue = Queue(maxsize=8)
cola_resultados: Queue = Queue(maxsize=20)
evento_cierre = threading.Event()

estado = {
    "reproduciendo": False,
    "pausado": False,
    "finalizado": False,
    "autogiro": False,
    "sesion": 0,
    "siguiente_tiempo_analisis": 0.0,
}

datos = {
    "tiempo": [],
    "punto_pca": [],
    "punto_crudo": [],
    "punto_suave": [],
    "tamanio": [],
    "actividad": [],
    "rms": [],
    "centroide": [],
    "dominante": [],
    "zcr": [],
    "flujo": [],
}


# ============================================================
# 7. ANÁLISIS DE CADA VENTANA
# ============================================================


def extraer_segmento(tiempo_segundos: float) -> np.ndarray:
    centro = int(tiempo_segundos * frecuencia_muestreo)
    mitad = VENTANA_ANALISIS // 2
    inicio = centro - mitad
    fin = inicio + VENTANA_ANALISIS

    relleno_izquierdo = max(0, -inicio)
    relleno_derecho = max(0, fin - len(audio))

    inicio_real = max(0, inicio)
    fin_real = min(len(audio), fin)
    segmento = audio[inicio_real:fin_real]

    if relleno_izquierdo or relleno_derecho:
        segmento = np.pad(
            segmento,
            (relleno_izquierdo, relleno_derecho),
            mode="constant",
        )

    return librosa.util.fix_length(
        segmento.astype(np.float32, copy=False),
        size=VENTANA_ANALISIS,
    )


def analizar_ventana(tiempo_segundos: float) -> dict:
    segmento = extraer_segmento(tiempo_segundos)

    mfcc = librosa.feature.mfcc(
        y=segmento,
        sr=frecuencia_muestreo,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        center=False,
    )
    chroma = librosa.feature.chroma_stft(
        y=segmento,
        sr=frecuencia_muestreo,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        center=False,
    )
    rms = librosa.feature.rms(
        y=segmento,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
        center=False,
    )
    centroide = librosa.feature.spectral_centroid(
        y=segmento,
        sr=frecuencia_muestreo,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        center=False,
    )
    ancho = librosa.feature.spectral_bandwidth(
        y=segmento,
        sr=frecuencia_muestreo,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        center=False,
    )
    rolloff = librosa.feature.spectral_rolloff(
        y=segmento,
        sr=frecuencia_muestreo,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        roll_percent=0.85,
        center=False,
    )
    zcr = librosa.feature.zero_crossing_rate(
        y=segmento,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
        center=False,
    )
    flatness = librosa.feature.spectral_flatness(
        y=segmento,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        center=False,
    )
    flujo = librosa.onset.onset_strength(
        y=segmento,
        sr=frecuencia_muestreo,
        hop_length=HOP_LENGTH,
        center=False,
    )

    vector = np.concatenate(
        [
            np.mean(mfcc, axis=1),
            np.mean(chroma, axis=1),
            [float(np.mean(rms))],
            [float(np.mean(centroide))],
            [float(np.mean(ancho))],
            [float(np.mean(rolloff))],
            [float(np.mean(zcr))],
            [float(np.mean(flatness))],
        ]
    )
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

    punto_pca = pca.transform(
        escalador.transform(vector.reshape(1, -1))
    )[0]
    punto_crudo = expandir_coordenadas_pca(punto_pca)

    valor_rms = float(np.mean(rms))
    valor_centroide = float(np.mean(centroide))
    valor_zcr = float(np.mean(zcr))
    valor_flujo = float(np.mean(flujo)) if flujo.size else 0.0

    # Frecuencia dominante mediante FFT.
    ventana_fft = np.hanning(len(segmento))
    espectro = np.abs(np.fft.rfft(segmento * ventana_fft))
    frecuencias_fft = np.fft.rfftfreq(
        len(segmento), d=1.0 / frecuencia_muestreo
    )

    if len(espectro) > 1:
        indice_pico = int(np.argmax(espectro[1:]) + 1)
        frecuencia_dominante = float(frecuencias_fft[indice_pico])
    else:
        frecuencia_dominante = 0.0

    energia_n = normalizar_por_rango(valor_rms, rms_inf, rms_sup)
    flujo_n = normalizar_por_rango(valor_flujo, flujo_inf, flujo_sup)
    centroide_n = normalizar_por_rango(
        valor_centroide, centroide_inf, centroide_sup
    )
    zcr_n = normalizar_por_rango(valor_zcr, zcr_inf, zcr_sup)

    # Tamaño del nodo basado en el sonido:
    # energía = volumen, flujo = cambio, centroide = brillo, zcr = transitorio.
    actividad = float(
        np.clip(
            0.52 * energia_n
            + 0.28 * flujo_n
            + 0.12 * centroide_n
            + 0.08 * zcr_n,
            0.0,
            1.0,
        )
    )

    tamanio = float(
        TAMANIO_NODO_MIN
        + (TAMANIO_NODO_MAX - TAMANIO_NODO_MIN) * actividad**1.32
    )

    return {
        "tiempo": float(tiempo_segundos),
        "punto_pca": punto_pca,
        "punto_crudo": punto_crudo,
        "tamanio": tamanio,
        "actividad": actividad,
        "rms": valor_rms,
        "centroide": valor_centroide,
        "dominante": frecuencia_dominante,
        "zcr": valor_zcr,
        "flujo": valor_flujo,
    }


def trabajador_analisis() -> None:
    while not evento_cierre.is_set():
        try:
            sesion, tiempo_objetivo = cola_solicitudes.get(timeout=0.15)
        except Empty:
            continue

        try:
            resultado = analizar_ventana(tiempo_objetivo)
            resultado["sesion"] = sesion

            try:
                cola_resultados.put(resultado, timeout=0.15)
            except Full:
                pass
        except Exception as error:
            try:
                cola_resultados.put(
                    {
                        "sesion": sesion,
                        "tiempo": tiempo_objetivo,
                        "error": str(error),
                    },
                    timeout=0.05,
                )
            except Full:
                pass
        finally:
            cola_solicitudes.task_done()


hilo_analisis = threading.Thread(
    target=trabajador_analisis,
    name="AnalizadorAcustico",
    daemon=True,
)
hilo_analisis.start()


# ============================================================
# 8. CREACIÓN DE LA INTERFAZ PROFESIONAL
# ============================================================

plt.style.use("dark_background")
plt.rcParams["toolbar"] = "None"
plt.rcParams["font.family"] = "DejaVu Sans"
figura = plt.figure(figsize=(18.0, 10.2), facecolor=COLOR_FONDO)

try:
    figura.canvas.manager.set_window_title(
        "Acoustic Lab Pro — Visualizador acústico 3D"
    )
except Exception:
    pass

# Iluminación ambiental muy tenue para dar profundidad sin distraer.
for x, y, radio, color, alpha in (
    (0.16, 0.83, 0.24, COLOR_ACENTO, 0.032),
    (0.82, 0.76, 0.27, COLOR_ACENTO_2, 0.035),
    (0.56, 0.09, 0.30, COLOR_ACENTO_3, 0.022),
):
    figura.add_artist(
        Circle(
            (x, y),
            radio,
            transform=figura.transFigure,
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
            zorder=-50,
        )
    )

# Bandas estructurales del dashboard.
figura.add_artist(
    Rectangle(
        (0.0, 0.915),
        1.0,
        0.085,
        transform=figura.transFigure,
        facecolor=COLOR_FONDO_2,
        edgecolor="none",
        zorder=-20,
    )
)
figura.add_artist(
    Rectangle(
        (0.0, 0.000),
        1.0,
        0.090,
        transform=figura.transFigure,
        facecolor="#070d1d",
        edgecolor="none",
        zorder=-20,
    )
)

# Línea superior multicolor que identifica la aplicación.
for x, ancho, color in (
    (0.000, 0.34, COLOR_ACENTO),
    (0.340, 0.33, COLOR_ACENTO_2),
    (0.670, 0.33, COLOR_ACENTO_3),
):
    figura.add_artist(
        Rectangle(
            (x, 0.996),
            ancho,
            0.004,
            transform=figura.transFigure,
            facecolor=color,
            edgecolor="none",
            zorder=30,
        )
    )

# Contenedores redondeados detrás de las áreas principales.
figura.add_artist(
    FancyBboxPatch(
        (0.025, 0.215),
        0.700,
        0.685,
        boxstyle="round,pad=0.008,rounding_size=0.018",
        transform=figura.transFigure,
        facecolor="#071022",
        edgecolor=COLOR_BORDE,
        linewidth=1.15,
        zorder=-10,
    )
)
figura.add_artist(
    FancyBboxPatch(
        (0.738, 0.215),
        0.237,
        0.685,
        boxstyle="round,pad=0.008,rounding_size=0.018",
        transform=figura.transFigure,
        facecolor=COLOR_PANEL,
        edgecolor=COLOR_BORDE,
        linewidth=1.15,
        zorder=-10,
    )
)

# Distribución equilibrada y centrada.
eje_3d = figura.add_axes([0.045, 0.245, 0.640, 0.615], projection="3d")
eje_color = figura.add_axes([0.700, 0.305, 0.010, 0.485])
eje_panel = figura.add_axes([0.748, 0.235, 0.217, 0.645])
eje_onda = figura.add_axes([0.050, 0.120, 0.915, 0.070])
eje_progreso = figura.add_axes([0.050, 0.091, 0.915, 0.014])

for eje_aux in (eje_panel, eje_onda, eje_progreso):
    eje_aux.set_facecolor(COLOR_PANEL)

# Encabezado de marca.
figura.text(
    0.035,
    0.965,
    "ACOUSTIC LAB  /  PRO",
    ha="left",
    va="center",
    fontsize=9.0,
    fontweight="bold",
    color=COLOR_ACENTO,
)
figura.text(
    0.035,
    0.938,
    "VISUALIZADOR ACÚSTICO ESPACIOTEMPORAL",
    ha="left",
    va="center",
    fontsize=17.5,
    fontweight="bold",
    color=COLOR_TEXTO,
)
figura.text(
    0.585,
    0.940,
    f"{RUTA_AUDIO.name}  /  40 MFCC  /  PCA  /  TIEMPO REAL",
    ha="center",
    va="center",
    fontsize=7.2,
    fontweight="bold",
    color=COLOR_SECUNDARIO,
)

# Insignias informativas del encabezado.
def crear_insignia(x: float, texto: str, color: str, ancho: float) -> None:
    figura.add_artist(
        FancyBboxPatch(
            (x, 0.934),
            ancho,
            0.032,
            boxstyle="round,pad=0.004,rounding_size=0.010",
            transform=figura.transFigure,
            facecolor=color,
            edgecolor="none",
            alpha=0.18,
            zorder=5,
        )
    )
    figura.text(
        x + ancho / 2,
        0.950,
        texto,
        ha="center",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        color=color,
        zorder=6,
    )


indicador_estado_header = Circle(
    (0.724, 0.950),
    0.0052,
    transform=figura.transFigure,
    facecolor=COLOR_ACENTO,
    edgecolor="#d1fae5",
    linewidth=0.7,
    zorder=8,
)
figura.add_artist(indicador_estado_header)
texto_estado_header = figura.text(
    0.735,
    0.950,
    "LISTO",
    ha="left",
    va="center",
    fontsize=6.5,
    fontweight="bold",
    color=COLOR_SECUNDARIO,
    zorder=8,
)

crear_insignia(0.785, RUTA_AUDIO.suffix.upper().replace(".", ""), COLOR_ACENTO, 0.052)
crear_insignia(0.845, f"{duracion:.1f} s", COLOR_ACENTO_2, 0.060)
crear_insignia(0.913, f"{frecuencia_muestreo / 1000:.1f} kHz", COLOR_ACENTO_3, 0.065)

texto_notificacion = figura.text(
    0.500,
    0.904,
    "",
    ha="center",
    va="center",
    fontsize=7.3,
    fontweight="bold",
    color=COLOR_TEXTO,
    visible=False,
    zorder=40,
    bbox={
        "boxstyle": "round,pad=0.55",
        "facecolor": "#111d38",
        "edgecolor": COLOR_BORDE,
        "linewidth": 0.9,
        "alpha": 0.97,
    },
)
temporizador_notificacion = None


def mostrar_notificacion(mensaje: str, color: str = COLOR_INFO) -> None:
    """Muestra un aviso breve dentro de la aplicación."""
    global temporizador_notificacion

    texto_notificacion.set_text(mensaje)
    texto_notificacion.set_color(color)
    texto_notificacion.set_visible(True)

    if temporizador_notificacion is not None:
        try:
            temporizador_notificacion.stop()
        except Exception:
            pass

    temporizador_notificacion = figura.canvas.new_timer(interval=2800)
    temporizador_notificacion.single_shot = True

    def ocultar() -> None:
        texto_notificacion.set_visible(False)
        figura.canvas.draw_idle()

    temporizador_notificacion.add_callback(ocultar)
    temporizador_notificacion.start()
    figura.canvas.draw_idle()


# ============================================================
# 9. GRÁFICO 3D VACÍO AL INICIO
# ============================================================

normalizacion_tiempo = Normalize(vmin=0.0, vmax=max(duracion, 0.01))
eje_3d.set_facecolor("#071022")

# Límites robustos: el manifold ocupa aproximadamente 95 % del volumen 3D.
LIMITES_BASE_X = limitar_eje(coordenadas_calibracion_suaves[:, 0])
LIMITES_BASE_Y = limitar_eje(coordenadas_calibracion_suaves[:, 1])
LIMITES_BASE_Z = limitar_eje(coordenadas_calibracion_suaves[:, 2])

eje_3d.set_xlim(LIMITES_BASE_X)
eje_3d.set_ylim(LIMITES_BASE_Y)
eje_3d.set_zlim(LIMITES_BASE_Z)

RANGOS_BASE_3D = np.array(
    [
        LIMITES_BASE_X[1] - LIMITES_BASE_X[0],
        LIMITES_BASE_Y[1] - LIMITES_BASE_Y[0],
        LIMITES_BASE_Z[1] - LIMITES_BASE_Z[0],
    ],
    dtype=float,
)

# Dos capas de línea: halo luminoso + línea principal nítida.
trayectoria_halo = Line3DCollection(
    [],
    cmap=MAPA_TIEMPO,
    norm=normalizacion_tiempo,
    linewidths=7.0,
    alpha=0.13,
)
trayectoria_color = Line3DCollection(
    [],
    cmap=MAPA_TIEMPO,
    norm=normalizacion_tiempo,
    linewidths=2.15,
    alpha=0.99,
)
eje_3d.add_collection3d(trayectoria_halo, autolim=False)
eje_3d.add_collection3d(trayectoria_color, autolim=False)

nube_nodos = eje_3d.scatter(
    [],
    [],
    [],
    c=[],
    s=[],
    cmap=MAPA_TIEMPO,
    norm=normalizacion_tiempo,
    alpha=0.78,
    linewidths=0.35,
    edgecolors="#dbeafe",
    depthshade=False,
)

puntero_actual = eje_3d.scatter(
    [],
    [],
    [],
    s=225,
    facecolors="none",
    edgecolors=COLOR_CURSOR,
    linewidths=2.3,
    alpha=0.98,
)

eje_3d.view_init(elev=27, azim=-56)
eje_3d.set_box_aspect((1.34, 1.23, 1.12))
eje_3d.set_xlabel("CP1 · ESTRUCTURA", labelpad=10, fontsize=8.2, color=COLOR_SECUNDARIO)
eje_3d.set_ylabel("CP2 · VARIACIÓN", labelpad=10, fontsize=8.2, color=COLOR_SECUNDARIO)
eje_3d.set_zlabel("CP3 · PROFUNDIDAD", labelpad=9, fontsize=8.2, color=COLOR_SECUNDARIO)
eje_3d.set_xticks([])
eje_3d.set_yticks([])
eje_3d.set_zticks([])
eje_3d.grid(False)

# Planos 3D con matices distintos para mejorar la profundidad.
eje_3d.xaxis.pane.set_facecolor((0.03, 0.20, 0.23, 0.46))
eje_3d.yaxis.pane.set_facecolor((0.20, 0.06, 0.27, 0.42))
eje_3d.zaxis.pane.set_facecolor((0.24, 0.15, 0.03, 0.34))
eje_3d.xaxis.pane.set_edgecolor((0.15, 0.38, 0.48, 0.50))
eje_3d.yaxis.pane.set_edgecolor((0.35, 0.18, 0.50, 0.45))
eje_3d.zaxis.pane.set_edgecolor((0.45, 0.30, 0.10, 0.38))

eje_3d.set_title(
    "MAPA ACÚSTICO TRIDIMENSIONAL",
    fontsize=11.0,
    fontweight="bold",
    color=COLOR_TEXTO,
    pad=9,
)
figura.text(
    0.365,
    0.855,
    "COLOR = TIEMPO     •     TAMAÑO = ACTIVIDAD     •     ARRASTRA PARA ROTAR",
    ha="center",
    va="center",
    fontsize=7.2,
    fontweight="bold",
    color=COLOR_SECUNDARIO,
)

mapeable = ScalarMappable(norm=normalizacion_tiempo, cmap=MAPA_TIEMPO)
mapeable.set_array(np.array([0.0, duracion]))
barra_color = figura.colorbar(mapeable, cax=eje_color)
barra_color.set_label(
    "LÍNEA DE TIEMPO",
    fontsize=7.3,
    fontweight="bold",
    color=COLOR_SECUNDARIO,
    labelpad=8,
)
barra_color.ax.tick_params(labelsize=7.0, colors=COLOR_SECUNDARIO, length=0)
barra_color.outline.set_edgecolor(COLOR_BORDE)
ticks_color = np.linspace(0.0, duracion, 6)
barra_color.set_ticks(ticks_color)
barra_color.set_ticklabels(
    [f"{x:.1f}s" if duracion < 10 else f"{x:.0f}s" for x in ticks_color]
)


# ============================================================
# 10. PANEL DE MÉTRICAS
# ============================================================

eje_panel.set_xlim(0, 1)
eje_panel.set_ylim(0, 1)
eje_panel.axis("off")

# Cabecera del panel lateral.
eje_panel.text(
    0.07,
    0.962,
    "PANEL DE TELEMETRÍA",
    fontsize=10.4,
    fontweight="bold",
    color=COLOR_TEXTO,
    va="center",
)
eje_panel.text(
    0.07,
    0.923,
    "Métricas acústicas sincronizadas",
    fontsize=7.0,
    color=COLOR_SECUNDARIO,
    va="center",
)
eje_panel.plot([0.07, 0.93], [0.895, 0.895], color=COLOR_BORDE, linewidth=1.0)


def crear_tarjeta(
    y: float,
    titulo: str,
    valor: str,
    detalle: str,
    color: str,
):
    # Sombra discreta.
    eje_panel.add_patch(
        FancyBboxPatch(
            (0.065, y - 0.006),
            0.875,
            0.112,
            boxstyle="round,pad=0.010,rounding_size=0.025",
            transform=eje_panel.transAxes,
            facecolor="#040817",
            edgecolor="none",
            alpha=0.75,
        )
    )
    caja = FancyBboxPatch(
        (0.055, y),
        0.875,
        0.112,
        boxstyle="round,pad=0.010,rounding_size=0.025",
        transform=eje_panel.transAxes,
        facecolor=COLOR_PANEL_2,
        edgecolor=COLOR_BORDE,
        linewidth=0.85,
    )
    eje_panel.add_patch(caja)
    eje_panel.add_patch(
        FancyBboxPatch(
            (0.055, y),
            0.018,
            0.112,
            boxstyle="round,pad=0.000,rounding_size=0.010",
            transform=eje_panel.transAxes,
            facecolor=color,
            edgecolor="none",
            alpha=0.95,
        )
    )
    eje_panel.text(
        0.105,
        y + 0.083,
        titulo,
        fontsize=6.7,
        fontweight="bold",
        color=COLOR_SECUNDARIO,
        va="center",
    )
    texto = eje_panel.text(
        0.105,
        y + 0.048,
        valor,
        fontsize=12.0,
        fontweight="bold",
        color=COLOR_TEXTO,
        va="center",
    )
    eje_panel.text(
        0.105,
        y + 0.018,
        detalle,
        fontsize=6.0,
        color=color,
        va="center",
    )
    return texto


texto_nodos = crear_tarjeta(
    0.755,
    "NODOS VISIBLES",
    "0",
    "Densidad adaptativa del recorrido",
    COLOR_ACENTO,
)
texto_rms = crear_tarjeta(
    0.620,
    "ENERGÍA RMS",
    "0.00000",
    "Intensidad sonora instantánea",
    "#38bdf8",
)
texto_centroide = crear_tarjeta(
    0.485,
    "CENTROIDE ESPECTRAL",
    "0 Hz",
    "Brillo y distribución de frecuencias",
    COLOR_ACENTO_2,
)
texto_dominante = crear_tarjeta(
    0.350,
    "FRECUENCIA DOMINANTE",
    "0 Hz",
    "Pico principal detectado",
    COLOR_ACENTO_3,
)
texto_actividad = crear_tarjeta(
    0.215,
    "ACTIVIDAD ACÚSTICA",
    "0 %",
    "Energía + flujo + brillo + transitorios",
    "#fb7185",
)

# Indicador visual horizontal de actividad.
barra_actividad_fondo = FancyBboxPatch(
    (0.075, 0.178),
    0.835,
    0.014,
    boxstyle="round,pad=0.001,rounding_size=0.007",
    transform=eje_panel.transAxes,
    facecolor="#202c45",
    edgecolor="none",
)
eje_panel.add_patch(barra_actividad_fondo)
barra_actividad_valor = Rectangle(
    (0.075, 0.178),
    0.0,
    0.014,
    transform=eje_panel.transAxes,
    facecolor="#fb7185",
    edgecolor="none",
    alpha=0.98,
)
eje_panel.add_patch(barra_actividad_valor)

estado_visual = eje_panel.text(
    0.5,
    0.118,
    "LISTO",
    ha="center",
    va="center",
    fontsize=8.4,
    fontweight="bold",
    color=COLOR_ACENTO,
    bbox={
        "boxstyle": "round,pad=0.55",
        "facecolor": "#082f2d",
        "edgecolor": "#1a8078",
        "linewidth": 1.0,
    },
)

texto_tiempo_panel = eje_panel.text(
    0.5,
    0.061,
    f"00:00.00  /  {formatear_tiempo(duracion)}",
    ha="center",
    va="center",
    fontsize=9.0,
    fontweight="bold",
    color=COLOR_TEXTO,
)
eje_panel.text(
    0.5,
    0.026,
    "RUEDA: ZOOM  •  DOBLE CLIC: RESTABLECER",
    ha="center",
    va="center",
    fontsize=5.8,
    color=COLOR_SECUNDARIO,
)


# ============================================================
# 11. FORMA DE ONDA Y PROGRESO
# ============================================================

max_puntos_onda = 6500
paso_onda = max(1, len(audio) // max_puntos_onda)
audio_onda = audio[::paso_onda]
tiempo_onda = np.arange(len(audio_onda)) * paso_onda / frecuencia_muestreo

# Onda completa tenue y recorrido activo brillante.
eje_onda.fill_between(
    tiempo_onda,
    audio_onda,
    0,
    color=COLOR_ACENTO_2,
    alpha=0.08,
    linewidth=0,
)
eje_onda.plot(
    tiempo_onda,
    audio_onda,
    linewidth=0.60,
    color="#67e8f9",
    alpha=0.34,
)
onda_activa, = eje_onda.plot(
    [],
    [],
    linewidth=1.20,
    color="#fbbf24",
    alpha=0.98,
)
cursor_onda = eje_onda.axvline(
    0.0,
    linewidth=1.45,
    color="white",
    alpha=0.96,
)

eje_onda.set_xlim(0.0, max(duracion, 0.01))
limite_amplitud = max(float(np.max(np.abs(audio_onda))) * 1.18, 0.1)
eje_onda.set_ylim(-limite_amplitud, limite_amplitud)
eje_onda.set_yticks([])
eje_onda.tick_params(axis="x", colors=COLOR_SECUNDARIO, labelsize=7, length=2)
for borde in eje_onda.spines.values():
    borde.set_color(COLOR_BORDE)
    borde.set_linewidth(0.8)
eje_onda.set_title(
    "FORMA DE ONDA  ·  POSICIÓN DE REPRODUCCIÓN",
    loc="left",
    fontsize=7.8,
    fontweight="bold",
    color=COLOR_TEXTO,
    pad=4,
)

eje_progreso.set_xlim(0, 1)
eje_progreso.set_ylim(0, 1)
eje_progreso.set_xticks([])
eje_progreso.set_yticks([])
for borde in eje_progreso.spines.values():
    borde.set_visible(False)

barra_fondo = FancyBboxPatch(
    (0.0, 0.12),
    1.0,
    0.76,
    boxstyle="round,pad=0.02,rounding_size=0.16",
    transform=eje_progreso.transAxes,
    facecolor="#1b2942",
    edgecolor=COLOR_BORDE,
    linewidth=0.65,
)
eje_progreso.add_patch(barra_fondo)

barra_relleno = Rectangle(
    (0.0, 0.12),
    0.0,
    0.76,
    transform=eje_progreso.transAxes,
    facecolor=COLOR_ACENTO,
    edgecolor="none",
    alpha=0.98,
)
eje_progreso.add_patch(barra_relleno)

texto_progreso = figura.text(
    0.965,
    0.103,
    "0 %",
    ha="right",
    va="center",
    fontsize=6.7,
    fontweight="bold",
    color=COLOR_ACENTO,
)


# ============================================================
# 12. BOTONES Y ÁREA DE CONTROL
# ============================================================

figura.text(
    0.035,
    0.078,
    "CENTRO DE CONTROL",
    fontsize=6.7,
    fontweight="bold",
    color=COLOR_SECUNDARIO,
    va="center",
)
figura.text(
    0.965,
    0.008,
    "ESPACIO: PLAY  •  P: PAUSA  •  R/ESC: DETENER  •  A: AUTOGIRO  •  0: CÁMARA",
    fontsize=5.2,
    color=COLOR_SECUNDARIO,
    ha="right",
    va="center",
)

posiciones_botones = [0.035, 0.153, 0.271, 0.389, 0.507, 0.625, 0.743]
ejes_botones = [
    figura.add_axes([x, 0.023, 0.105, 0.045])
    for x in posiciones_botones
]
(
    ax_open,
    ax_play,
    ax_pause,
    ax_stop,
    ax_save,
    ax_camera,
    ax_rotate,
) = ejes_botones

boton_open = Button(ax_open, "ABRIR AUDIO")
boton_play = Button(ax_play, "REPRODUCIR")
boton_pause = Button(ax_pause, "PAUSAR")
boton_stop = Button(ax_stop, "DETENER")
boton_save = Button(ax_save, "EXPORTAR")
boton_camera = Button(ax_camera, "CÁMARA")
boton_autogiro = Button(ax_rotate, "AUTOGIRO: OFF")


def estilizar_boton(
    boton: Button,
    color: str,
    color_hover: str,
    borde: str = "#ffffff",
) -> None:
    boton.ax.set_facecolor(color)
    boton.ax.patch.set_edgecolor(borde)
    boton.ax.patch.set_alpha(0.94)
    boton.ax.patch.set_linewidth(0.55)
    boton.label.set_color("white")
    boton.label.set_fontsize(7.0)
    boton.label.set_fontweight("bold")
    boton.color = color
    boton.hovercolor = color_hover
    boton.ax.set_xticks([])
    boton.ax.set_yticks([])
    for borde_ax in boton.ax.spines.values():
        borde_ax.set_visible(False)


estilizar_boton(boton_open, "#164e63", "#0891b2")
estilizar_boton(boton_play, "#0f766e", "#14b8a6")
estilizar_boton(boton_pause, "#334155", "#475569")
estilizar_boton(boton_stop, "#991b1b", "#dc2626")
estilizar_boton(boton_save, "#5b21b6", "#7c3aed")
estilizar_boton(boton_camera, "#1e3a8a", "#2563eb")
estilizar_boton(boton_autogiro, "#3f3f46", "#52525b")

figura.text(
    0.885,
    0.077,
    "VOLUMEN",
    fontsize=6.3,
    fontweight="bold",
    color=COLOR_SECUNDARIO,
    ha="left",
    va="center",
)
ax_volumen = figura.add_axes([0.875, 0.037, 0.088, 0.016], facecolor="none")
control_volumen = Slider(
    ax=ax_volumen,
    label="",
    valmin=0.0,
    valmax=1.0,
    valinit=VOLUMEN_INICIAL,
    valstep=0.01,
    color=COLOR_ACENTO,
)
try:
    control_volumen.track.set_facecolor("#1b2942")
    control_volumen.poly.set_alpha(0.98)
except Exception:
    pass
control_volumen.valtext.set_color(COLOR_TEXTO)
control_volumen.valtext.set_fontsize(6.2)
control_volumen.valtext.set_text(f"{int(VOLUMEN_INICIAL * 100)}%")
for borde in ax_volumen.spines.values():
    borde.set_visible(False)


def cambiar_volumen(valor: float) -> None:
    volumen = float(np.clip(valor, 0.0, 1.0))
    pygame.mixer.music.set_volume(volumen)
    control_volumen.valtext.set_text(f"{int(round(volumen * 100))}%")
    figura.canvas.draw_idle()


control_volumen.on_changed(cambiar_volumen)


def seleccionar_audio(_evento=None) -> None:
    """Abre un selector y reinicia la aplicación con el archivo elegido."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        raiz = tk.Tk()
        raiz.withdraw()
        try:
            raiz.attributes("-topmost", True)
        except Exception:
            pass

        seleccionado = filedialog.askopenfilename(
            parent=raiz,
            title="Seleccionar audio WAV o MP3",
            filetypes=[
                ("Archivos de audio", "*.wav *.mp3"),
                ("WAV", "*.wav"),
                ("MP3", "*.mp3"),
            ],
        )
        raiz.destroy()
    except Exception as error:
        mostrar_notificacion(f"No se pudo abrir el selector: {error}", COLOR_PELIGRO)
        return

    if not seleccionado:
        return

    ruta_nueva = Path(seleccionado)
    if ruta_nueva.suffix.lower() not in EXTENSIONES_AUDIO_ADMITIDAS:
        mostrar_notificacion("Formato no admitido. Usa WAV o MP3.", COLOR_PELIGRO)
        return

    try:
        pygame.mixer.music.stop()
        pygame.mixer.quit()
    except Exception:
        pass

    evento_cierre.set()
    script_actual = str(Path(__file__).resolve())
    os.execv(sys.executable, [sys.executable, script_actual, str(ruta_nueva.resolve())])


def alternar_autogiro(_evento=None) -> None:
    estado["autogiro"] = not estado["autogiro"]
    if estado["autogiro"]:
        boton_autogiro.label.set_text("AUTOGIRO: ON")
        boton_autogiro.color = "#9a3412"
        boton_autogiro.hovercolor = "#ea580c"
        boton_autogiro.ax.set_facecolor("#9a3412")
        mostrar_notificacion("Autogiro 3D activado", COLOR_ACENTO_3)
    else:
        boton_autogiro.label.set_text("AUTOGIRO: OFF")
        boton_autogiro.color = "#3f3f46"
        boton_autogiro.hovercolor = "#52525b"
        boton_autogiro.ax.set_facecolor("#3f3f46")
        mostrar_notificacion("Autogiro 3D desactivado", COLOR_SECUNDARIO)
    figura.canvas.draw_idle()


def restablecer_camara(_evento=None) -> None:
    eje_3d.view_init(elev=27, azim=-56)
    eje_3d.set_xlim3d(LIMITES_BASE_X)
    eje_3d.set_ylim3d(LIMITES_BASE_Y)
    eje_3d.set_zlim3d(LIMITES_BASE_Z)
    mostrar_notificacion("Cámara 3D restablecida", COLOR_INFO)
    figura.canvas.draw_idle()


# ============================================================
# 13. LIMPIEZA Y GUARDADO
# ============================================================


def limpiar_visualizacion() -> None:
    for lista in datos.values():
        lista.clear()

    trayectoria_halo.set_segments([])
    trayectoria_halo.set_array(np.array([]))
    trayectoria_color.set_segments([])
    trayectoria_color.set_array(np.array([]))

    nube_nodos._offsets3d = (np.array([]), np.array([]), np.array([]))
    nube_nodos.set_sizes(np.array([]))
    nube_nodos.set_array(np.array([]))

    puntero_actual._offsets3d = (np.array([]), np.array([]), np.array([]))
    puntero_actual.set_sizes(np.array([210.0]))

    onda_activa.set_data([], [])
    cursor_onda.set_xdata([0.0, 0.0])
    barra_relleno.set_width(0.0)
    barra_actividad_valor.set_width(0.0)
    barra_actividad_valor.set_facecolor(COLOR_ACENTO)
    texto_progreso.set_text("0 %")

    texto_nodos.set_text("0")
    texto_rms.set_text("0.00000")
    texto_centroide.set_text("0 Hz")
    texto_dominante.set_text("0 Hz")
    texto_actividad.set_text("0 %")
    texto_tiempo_panel.set_text(
        f"00:00.00 / {formatear_tiempo(duracion)}"
    )


def guardar_resultados() -> None:
    figura.savefig(
        RUTA_CAPTURA,
        dpi=220,
        bbox_inches="tight",
        facecolor=figura.get_facecolor(),
    )

    if datos["tiempo"]:
        pca_original = np.asarray(datos["punto_pca"], dtype=float)
        crudos = np.asarray(datos["punto_crudo"], dtype=float)
        suaves = np.asarray(datos["punto_suave"], dtype=float)
        matriz = np.column_stack(
            [
                np.asarray(datos["tiempo"]),
                pca_original[:, 0],
                pca_original[:, 1],
                pca_original[:, 2],
                crudos[:, 0],
                crudos[:, 1],
                crudos[:, 2],
                suaves[:, 0],
                suaves[:, 1],
                suaves[:, 2],
                np.asarray(datos["rms"]),
                np.asarray(datos["centroide"]),
                np.asarray(datos["dominante"]),
                np.asarray(datos["zcr"]),
                np.asarray(datos["flujo"]),
                np.asarray(datos["actividad"]),
                np.asarray(datos["tamanio"]),
            ]
        )

        np.savetxt(
            RUTA_CSV,
            matriz,
            delimiter=",",
            header=(
                "tiempo_segundos,pca1_original,pca2_original,pca3_original,"
                "visual1_expandido,visual2_expandido,visual3_expandido,"
                "visual1_suave,visual2_suave,visual3_suave,rms,centroide_hz,"
                "frecuencia_dominante_hz,zcr,flujo_espectral,"
                "actividad_normalizada,tamanio_nodo"
            ),
            comments="",
            fmt="%.7f",
        )

    print(f"Captura guardada: {RUTA_CAPTURA}")
    if datos["tiempo"]:
        print(f"Datos guardados: {RUTA_CSV}")


# ============================================================
# 14. CONTROLES DE REPRODUCCIÓN
# ============================================================


def establecer_estado_visual(
    texto: str,
    color_texto: str,
    color_fondo: str,
    color_borde: str,
) -> None:
    """Actualiza de manera consistente los indicadores de estado."""
    estado_visual.set_text(texto)
    estado_visual.set_color(color_texto)
    texto_estado_header.set_text(texto)
    indicador_estado_header.set_facecolor(color_texto)
    estado_visual.set_bbox(
        {
            "boxstyle": "round,pad=0.55",
            "facecolor": color_fondo,
            "edgecolor": color_borde,
            "linewidth": 1.0,
        }
    )


def preparar_sesion() -> None:
    estado["sesion"] += 1
    estado["siguiente_tiempo_analisis"] = 0.0
    limpiar_cola(cola_solicitudes)
    limpiar_cola(cola_resultados)
    limpiar_visualizacion()


def reproducir(_evento=None) -> None:
    if estado["pausado"]:
        pygame.mixer.music.unpause()
        estado["pausado"] = False
        estado["reproduciendo"] = True
        estado["finalizado"] = False
    elif not pygame.mixer.music.get_busy():
        preparar_sesion()
        pygame.mixer.music.stop()
        pygame.mixer.music.play()
        estado["reproduciendo"] = True
        estado["pausado"] = False
        estado["finalizado"] = False

    establecer_estado_visual(
        "ANALIZANDO", "#fde68a", "#3a2d08", "#d49a14"
    )
    figura.canvas.draw_idle()


def pausar(_evento=None) -> None:
    if estado["reproduciendo"] and not estado["pausado"]:
        pygame.mixer.music.pause()
        estado["pausado"] = True
        establecer_estado_visual(
            "PAUSADO", "#dbeafe", "#1e293b", "#64748b"
        )
        figura.canvas.draw_idle()


def detener(_evento=None) -> None:
    pygame.mixer.music.stop()
    estado["reproduciendo"] = False
    estado["pausado"] = False
    estado["finalizado"] = False
    preparar_sesion()
    establecer_estado_visual(
        "LISTO", COLOR_ACENTO, "#082f2d", "#1a8078"
    )
    figura.canvas.draw_idle()


def guardar(_evento=None) -> None:
    guardar_resultados()
    establecer_estado_visual(
        "EXPORTADO", "#ede9fe", "#2e1065", "#7c3aed"
    )
    mostrar_notificacion(
        f"Exportado: {RUTA_CAPTURA.name}"
        + (f" + {RUTA_CSV.name}" if datos["tiempo"] else ""),
        "#c4b5fd",
    )
    figura.canvas.draw_idle()


boton_open.on_clicked(seleccionar_audio)
boton_play.on_clicked(reproducir)
boton_pause.on_clicked(pausar)
boton_stop.on_clicked(detener)
boton_save.on_clicked(guardar)
boton_camera.on_clicked(restablecer_camara)
boton_autogiro.on_clicked(alternar_autogiro)


# ============================================================
# 15. RECEPCIÓN DE RESULTADOS Y SUAVIZADO
# ============================================================


def incorporar_resultados() -> bool:
    hubo_cambios = False

    while True:
        try:
            resultado = cola_resultados.get_nowait()
        except Empty:
            break

        if resultado.get("sesion") != estado["sesion"]:
            continue

        if "error" in resultado:
            print(
                f"Aviso en {resultado.get('tiempo', 0.0):.2f}s: "
                f"{resultado['error']}"
            )
            continue

        tiempo = float(resultado["tiempo"])
        if datos["tiempo"] and tiempo <= datos["tiempo"][-1]:
            continue

        punto_pca = np.asarray(resultado["punto_pca"], dtype=float)
        punto_crudo = np.asarray(resultado["punto_crudo"], dtype=float)
        if datos["punto_suave"]:
            anterior = np.asarray(datos["punto_suave"][-1], dtype=float)
            punto_suave = (
                FACTOR_SUAVIZADO * punto_crudo
                + (1.0 - FACTOR_SUAVIZADO) * anterior
            )
        else:
            punto_suave = punto_crudo.copy()

        datos["tiempo"].append(tiempo)
        datos["punto_pca"].append(punto_pca)
        datos["punto_crudo"].append(punto_crudo)
        datos["punto_suave"].append(punto_suave)
        datos["tamanio"].append(resultado["tamanio"])
        datos["actividad"].append(resultado["actividad"])
        datos["rms"].append(resultado["rms"])
        datos["centroide"].append(resultado["centroide"])
        datos["dominante"].append(resultado["dominante"])
        datos["zcr"].append(resultado["zcr"])
        datos["flujo"].append(resultado["flujo"])

        if len(datos["tiempo"]) > MAXIMO_NODOS_MEMORIA:
            for lista in datos.values():
                lista.pop(0)

        hubo_cambios = True

    return hubo_cambios


def actualizar_recorrido_3d() -> None:
    if not datos["tiempo"]:
        return

    tiempos_todos = np.asarray(datos["tiempo"], dtype=float)
    puntos_todos = np.asarray(datos["punto_suave"], dtype=float)
    tamanios_todos = np.asarray(datos["tamanio"], dtype=float)

    # Ventana móvil: mantiene claridad sin perder sincronización.
    inicio_visible = max(0, len(tiempos_todos) - MAXIMO_NODOS_VISIBLES)
    tiempos_visibles = tiempos_todos[inicio_visible:]
    puntos_visibles = puntos_todos[inicio_visible:]
    tamanios_visibles = tamanios_todos[inicio_visible:]

    nube_nodos._offsets3d = (
        puntos_visibles[:, 0],
        puntos_visibles[:, 1],
        puntos_visibles[:, 2],
    )
    nube_nodos.set_sizes(tamanios_visibles)
    nube_nodos.set_array(tiempos_visibles)

    if len(puntos_visibles) > 1:
        # Paso adaptativo: conserva la forma general y evita una maraña.
        salto = max(
            1,
            int(np.ceil((len(puntos_visibles) - 1) / MAXIMO_SEGMENTOS_VISIBLES)),
        )
        indices_linea = np.arange(0, len(puntos_visibles), salto, dtype=int)
        if indices_linea[-1] != len(puntos_visibles) - 1:
            indices_linea = np.append(indices_linea, len(puntos_visibles) - 1)

        puntos_linea = puntos_visibles[indices_linea]
        tiempos_linea = tiempos_visibles[indices_linea]

        segmentos = np.stack(
            [puntos_linea[:-1], puntos_linea[1:]],
            axis=1,
        )
        tiempo_segmentos = (
            tiempos_linea[:-1] + tiempos_linea[1:]
        ) / 2.0

        trayectoria_halo.set_segments(segmentos)
        trayectoria_halo.set_array(tiempo_segmentos)
        trayectoria_color.set_segments(segmentos)
        trayectoria_color.set_array(tiempo_segmentos)
    else:
        trayectoria_halo.set_segments([])
        trayectoria_halo.set_array(np.array([]))
        trayectoria_color.set_segments([])
        trayectoria_color.set_array(np.array([]))

    ultimo = puntos_visibles[-1]
    puntero_actual._offsets3d = (
        np.array([ultimo[0]]),
        np.array([ultimo[1]]),
        np.array([ultimo[2]]),
    )
    puntero_actual.set_sizes(
        np.array([max(180.0, tamanios_visibles[-1] * 1.75)])
    )

    texto_nodos.set_text(f"{len(puntos_visibles):,}")
    texto_rms.set_text(f"{datos['rms'][-1]:.5f}")
    texto_centroide.set_text(f"{datos['centroide'][-1]:,.0f} Hz")
    texto_dominante.set_text(f"{datos['dominante'][-1]:,.0f} Hz")
    actividad_actual = float(datos["actividad"][-1])
    texto_actividad.set_text(f"{actividad_actual * 100:.0f} %")
    barra_actividad_valor.set_width(0.835 * actividad_actual)
    barra_actividad_valor.set_facecolor(MAPA_ACTIVIDAD(actividad_actual))


# ============================================================
# 16. ANIMACIÓN Y SINCRONIZACIÓN
# ============================================================


def programar_ventanas_hasta(tiempo_actual: float) -> None:
    # En cada actualización se programan como máximo cuatro ventanas para
    # evitar que un equipo lento acumule una cola demasiado grande.
    programadas = 0

    while (
        estado["siguiente_tiempo_analisis"] <= tiempo_actual
        and estado["siguiente_tiempo_analisis"] <= duracion
        and programadas < 4
    ):
        tiempo_objetivo = float(estado["siguiente_tiempo_analisis"])
        try:
            cola_solicitudes.put_nowait((estado["sesion"], tiempo_objetivo))
        except Full:
            break

        estado["siguiente_tiempo_analisis"] += PASO_ANALISIS_SEGUNDOS
        programadas += 1


def actualizar_animacion(_numero):
    if incorporar_resultados():
        actualizar_recorrido_3d()

    if estado["autogiro"] and datos["tiempo"]:
        eje_3d.azim = (float(eje_3d.azim) + VELOCIDAD_AUTOGIRO) % 360.0

    if datos["tiempo"]:
        pulso = 1.0 + 0.09 * np.sin(float(_numero) * 0.28)
        tamanio_base = max(180.0, float(datos["tamanio"][-1]) * 1.75)
        puntero_actual.set_sizes(np.array([tamanio_base * pulso]))

    if not estado["reproduciendo"] and not estado["finalizado"]:
        return (
            trayectoria_halo,
            trayectoria_color,
            nube_nodos,
            puntero_actual,
            onda_activa,
            cursor_onda,
            barra_relleno,
        )

    esta_sonando = pygame.mixer.music.get_busy()

    if estado["reproduciendo"] and not estado["pausado"] and not esta_sonando:
        tiempo_actual = duracion
        programar_ventanas_hasta(tiempo_actual)
        estado["reproduciendo"] = False
        estado["finalizado"] = True
        establecer_estado_visual(
            "FINALIZADO", "#ede9fe", "#2e1065", "#7c3aed"
        )
    else:
        posicion_ms = pygame.mixer.music.get_pos()
        tiempo_actual = max(0.0, posicion_ms / 1000.0)
        tiempo_actual = min(tiempo_actual, duracion)

    if estado["reproduciendo"] and not estado["pausado"]:
        programar_ventanas_hasta(tiempo_actual)

    indice_onda = np.searchsorted(tiempo_onda, tiempo_actual, side="right")
    onda_activa.set_data(tiempo_onda[:indice_onda], audio_onda[:indice_onda])
    cursor_onda.set_xdata([tiempo_actual, tiempo_actual])

    progreso = float(np.clip(tiempo_actual / max(duracion, 0.01), 0.0, 1.0))
    barra_relleno.set_width(progreso)
    texto_progreso.set_text(f"{progreso * 100:.0f} %")
    texto_tiempo_panel.set_text(
        f"{formatear_tiempo(tiempo_actual)} / {formatear_tiempo(duracion)}"
    )

    return (
        trayectoria_halo,
        trayectoria_color,
        nube_nodos,
        puntero_actual,
        onda_activa,
        cursor_onda,
        barra_relleno,
    )


# ============================================================
# 17. ZOOM 3D CON RUEDA DEL MOUSE
# ============================================================


def aplicar_zoom_3d(factor: float) -> None:
    """Acerca o aleja la cámara reduciendo/ampliando los límites 3D."""
    limites_actuales = (
        eje_3d.get_xlim3d(),
        eje_3d.get_ylim3d(),
        eje_3d.get_zlim3d(),
    )

    nuevos_limites = []
    for indice, (minimo, maximo) in enumerate(limites_actuales):
        centro = (minimo + maximo) / 2.0
        rango_actual = maximo - minimo
        rango_nuevo = rango_actual * factor
        rango_nuevo = float(
            np.clip(
                rango_nuevo,
                RANGOS_BASE_3D[indice] * ZOOM_MINIMO_RELATIVO,
                RANGOS_BASE_3D[indice] * ZOOM_MAXIMO_RELATIVO,
            )
        )
        mitad = rango_nuevo / 2.0
        nuevos_limites.append((centro - mitad, centro + mitad))

    eje_3d.set_xlim3d(nuevos_limites[0])
    eje_3d.set_ylim3d(nuevos_limites[1])
    eje_3d.set_zlim3d(nuevos_limites[2])
    figura.canvas.draw_idle()


def manejar_scroll(evento) -> None:
    if evento.inaxes is not eje_3d:
        return

    # En Matplotlib: rueda arriba = acercar; rueda abajo = alejar.
    if evento.button == "up":
        aplicar_zoom_3d(FACTOR_ZOOM_ENTRADA)
    elif evento.button == "down":
        aplicar_zoom_3d(FACTOR_ZOOM_SALIDA)


def restablecer_zoom() -> None:
    eje_3d.set_xlim3d(LIMITES_BASE_X)
    eje_3d.set_ylim3d(LIMITES_BASE_Y)
    eje_3d.set_zlim3d(LIMITES_BASE_Z)
    figura.canvas.draw_idle()


def manejar_clic(evento) -> None:
    if evento.inaxes is eje_3d and getattr(evento, "dblclick", False):
        restablecer_camara()


# ============================================================
# 18. ATAJOS, MAXIMIZADO Y CIERRE
# ============================================================


def manejar_tecla(evento) -> None:
    tecla = (evento.key or "").lower()
    if tecla in {" ", "space"}:
        reproducir()
    elif tecla == "p":
        pausar()
    elif tecla in {"r", "escape"}:
        detener()
    elif tecla == "s":
        guardar()
    elif tecla == "a":
        alternar_autogiro()
    elif tecla == "o":
        seleccionar_audio()
    elif tecla in {"0", "home"}:
        restablecer_camara()


def maximizar_ventana() -> None:
    """Maximiza únicamente la ventana de esta figura, sin crear figuras nuevas."""
    try:
        administrador = figura.canvas.manager
        ventana = getattr(administrador, "window", None)
        if ventana is None:
            return
        if hasattr(ventana, "showMaximized"):
            ventana.showMaximized()
        elif hasattr(ventana, "state"):
            ventana.state("zoomed")
        elif hasattr(ventana, "wm_state"):
            ventana.wm_state("zoomed")
    except Exception:
        pass


def cerrar_ventana(_evento) -> None:
    """Detiene audio, animación y temporizadores antes de cerrar."""
    evento_cierre.set()
    estado["reproduciendo"] = False
    estado["pausado"] = False

    try:
        animacion.event_source.stop()
    except Exception:
        pass

    try:
        temporizador_maximizar.stop()
    except Exception:
        pass

    try:
        if temporizador_notificacion is not None:
            temporizador_notificacion.stop()
    except Exception:
        pass

    try:
        pygame.mixer.music.stop()
        pygame.mixer.quit()
    except Exception:
        pass


figura.canvas.mpl_connect("key_press_event", manejar_tecla)
figura.canvas.mpl_connect("scroll_event", manejar_scroll)
figura.canvas.mpl_connect("button_press_event", manejar_clic)
figura.canvas.mpl_connect("close_event", cerrar_ventana)

animacion = FuncAnimation(
    figura,
    actualizar_animacion,
    interval=INTERVALO_INTERFAZ_MS,
    cache_frame_data=False,
    blit=False,
)

temporizador_maximizar = figura.canvas.new_timer(interval=250)
# Ejecutarlo una sola vez. Si se repite después de cerrar, puede abrir figuras vacías.
temporizador_maximizar.single_shot = True
temporizador_maximizar.add_callback(maximizar_ventana)
temporizador_maximizar.start()

print("\n" + "=" * 76)
plt.show()