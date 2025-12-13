# agente.py
import os
import re
from collections import Counter
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import joblib

# ================================
# IMPORTS DEL PROYECTO (MANUAL)
# ================================

# K-Means jerárquico MANUAL (mismas salidas que antes: (clase_final, info))
from src.predecir_manual import predecir

# Bayes (igual que antes)
from src.bayes import estimar_posterior_secuencial

# Audio: mismo pipeline de extracción
from src.features_audio import extract_features_from_file
from src.preprocess_audio import preprocess_audio_file


# ================================
# CONFIGURACIÓN
# ================================

# Imágenes (10 piezas)
CARPETA_MUESTRA_IMG = "data/muestras"

# Audios numerados (1.wav, 2.wav, 3.wav, ...)
CARPETA_MUESTRA_AUDIO = "data_voz/muestras_audio"

# Temporal para preprocesado
TMP_PREPROCESSED = "data_voz/tmp_preprocessed.wav"

# Modelos MANUALES de voz
MODEL_PATH         = "data_voz/models_manual/knn_model.pkl"
SCALER_PATH        = "data_voz/models_manual/scaler.pkl"
FEATURE_NAMES_PATH = "data_voz/models_manual/feature_names.pkl"

SAMPLE_RATE = 16000
N_MFCC      = 13


# ================================
# UTILIDADES
# ================================

def _extraer_numero(nombre_archivo: str) -> int | None:
    """
    Devuelve el número del filename si es del estilo '12.wav' o '12_algo.wav'.
    Para tu caso ideal '12.wav'. Si no encuentra número, devuelve None.
    """
    base = os.path.splitext(nombre_archivo)[0]
    m = re.match(r"^\s*(\d+)", base)
    if not m:
        return None
    return int(m.group(1))


def listar_audios_ordenados(carpeta: str):
    """
    Lista los .wav de la carpeta y los ordena por número ascendente.
    Si hay audios sin número, los deja al final (orden alfabético).
    """
    if not os.path.isdir(carpeta):
        raise FileNotFoundError(f"No existe la carpeta de audios: {carpeta}")

    wavs = [f for f in os.listdir(carpeta) if f.lower().endswith(".wav")]
    if not wavs:
        raise FileNotFoundError(f"No hay .wav en: {carpeta}")

    con_num = []
    sin_num = []

    for f in wavs:
        n = _extraer_numero(f)
        if n is None:
            sin_num.append(f)
        else:
            con_num.append((n, f))

    con_num.sort(key=lambda t: t[0])
    sin_num.sort()

    ordenados = [f for _, f in con_num] + sin_num
    return [os.path.join(carpeta, f) for f in ordenados]


# ================================
# VISIÓN (IMÁGENES)
# ================================

def clasificar_muestra_imagenes(carpeta_muestra: str):
    piezas = []

    if not os.path.isdir(carpeta_muestra):
        print(f"[ERROR] La carpeta de imágenes no existe: {carpeta_muestra}")
        return piezas

    archivos = sorted(os.listdir(carpeta_muestra))
    if not archivos:
        print(f"[ADVERTENCIA] La carpeta {carpeta_muestra} está vacía.")
        return piezas

    print("\n=== CLASIFICACIÓN DE IMÁGENES (MANUAL) ===")
    for nombre_archivo in archivos:
        if not nombre_archivo.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        ruta_img = os.path.join(carpeta_muestra, nombre_archivo)
        try:
            clase, _info = predecir(ruta_img)
            piezas.append(clase)
            print(f"[VISION] {nombre_archivo} -> {clase}")
        except Exception as e:
            print(f"[ERROR] Clasificando {nombre_archivo}: {e}")

    return piezas


# ================================
# AUDIO (VOZ)
# ================================

def predecir_comando_desde_wav(
    input_wav: str,
    model_path: str = MODEL_PATH,
    scaler_path: str = SCALER_PATH,
    feature_names_path: str = FEATURE_NAMES_PATH,
    tmp_preprocessed: str = TMP_PREPROCESSED,
    sample_rate: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
):
    """
    Devuelve (comando, proba_dict).
    Usa KNN MANUAL entrenado en data_voz/models_manual.
    """
    if not os.path.exists(input_wav):
        raise FileNotFoundError(f"Archivo de audio no encontrado: {input_wav}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler no encontrado: {scaler_path}")
    if not os.path.exists(feature_names_path):
        raise FileNotFoundError(f"feature_names no encontrado: {feature_names_path}")

    # 1) Preprocesar
    preprocess_audio_file(
        input_path=input_wav,
        output_path=tmp_preprocessed,
        sample_rate=sample_rate,
    )

    # 2) Features
    feats = extract_features_from_file(
        tmp_preprocessed,
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
    )

    # 3) Orden columnas
    feature_names = joblib.load(feature_names_path)
    x_vec = np.array([feats[name] for name in feature_names], dtype=float).reshape(1, -1)

    # 4) Escalar + predecir
    scaler = joblib.load(scaler_path)
    knn = joblib.load(model_path)

    x_scaled = scaler.transform(x_vec)
    predicted_label = knn.predict(x_scaled)[0]

    proba_dict = None
    if hasattr(knn, "predict_proba") and hasattr(knn, "classes_"):
        proba = knn.predict_proba(x_scaled)[0]
        proba_dict = {str(c): float(p) for c, p in zip(knn.classes_, proba)}

    return str(predicted_label).strip().lower(), proba_dict


# ================================
# ACCIONES (CONTAR / PROPORCION / SALIR)
# ================================

def mostrar_conteo_piezas(piezas):
    if not piezas:
        print("[ADVERTENCIA] No hay piezas para contar.")
        return

    conteo = Counter(piezas)
    print("\n=== CONTEO DE PIEZAS EN LA MUESTRA ===")
    for tipo, cantidad in conteo.items():
        print(f"- {tipo}: {cantidad}")
    print("======================================\n")


def mostrar_estimacion_bayes(piezas):
    if not piezas:
        print("[ADVERTENCIA] No hay piezas en la muestra para estimar la caja.")
        return

    posterior_final, _historial = estimar_posterior_secuencial(piezas)
    caja_mas_prob = max(posterior_final, key=posterior_final.get)

    print("\n=== ESTIMACIÓN BAYESIANA DE LA CAJA DE ORIGEN ===")
    for caja, p in posterior_final.items():
        print(f"- Caja {caja}: {p:.4f}")
    print(f"\nCaja más probable: {caja_mas_prob}")
    print("=================================================\n")


def ejecutar_accion(piezas, comando):
    """
    Devuelve True si hay que salir.
    """
    if comando == "contar":
        mostrar_conteo_piezas(piezas)
        return False

    if comando == "proporcion":
        mostrar_estimacion_bayes(piezas)
        return False

    if comando == "salir":
        print("\n[INFO] Comando 'salir' recibido. Mostrando resumen final...\n")
        mostrar_conteo_piezas(piezas)
        mostrar_estimacion_bayes(piezas)
        print("[INFO] Finalizando el programa.")
        return True

    print(f"[WARN] Comando no reconocido: '{comando}' (esperaba contar/proporcion/salir)")
    return False


# ================================
# MAIN
# ================================

def main():
    print("==============================================")
    print("  AGENTE INTELIGENTE (MANUAL) - AUDIOS EN LISTA")
    print("==============================================\n")

    # 1) Clasificar piezas (una sola vez)
    print(f"[INFO] Clasificando imágenes en: {CARPETA_MUESTRA_IMG}")
    piezas = clasificar_muestra_imagenes(CARPETA_MUESTRA_IMG)

    if not piezas:
        print("[ERROR] No se obtuvieron piezas de la muestra. Saliendo.")
        return

    print(f"\n[INFO] Muestra clasificada. Total de piezas: {len(piezas)}")
    print(f"[INFO] Etiquetas obtenidas: {piezas}\n")

    # 2) Preparar lista de audios numerados
    try:
        lista_audios = listar_audios_ordenados(CARPETA_MUESTRA_AUDIO)
    except Exception as e:
        print(f"[ERROR] No se pudo preparar la lista de audios: {e}")
        return

    print("=== LISTA DE AUDIOS (EN ORDEN) ===")
    for i, p in enumerate(lista_audios, start=1):
        print(f"{i:02d}) {os.path.basename(p)}")
    print("=================================\n")

    # 3) Procesar audios uno por uno al presionar ENTER
    for idx, audio_path in enumerate(lista_audios, start=1):
        print(f"\n[INFO] Siguiente audio ({idx}/{len(lista_audios)}): {os.path.basename(audio_path)}")
        input("Audio listo → presione ENTER para procesar... ")

        try:
            comando, proba = predecir_comando_desde_wav(audio_path)
        except Exception as e:
            print(f"[ERROR] Falló el reconocimiento de voz para {audio_path}: {e}")
            continue

        print(f"[AUDIO] Comando reconocido: '{comando}'")
        if proba:
            print("[AUDIO] Probabilidades:")
            for cls, p in sorted(proba.items(), key=lambda t: t[1], reverse=True):
                print(f"  {cls}: {p:.4f}")

        terminar = ejecutar_accion(piezas, comando)
        if terminar:
            break

    print("\n[FIN] Programa finalizado.")


if __name__ == "__main__":
    main()
