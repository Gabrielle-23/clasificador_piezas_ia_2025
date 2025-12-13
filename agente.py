# agente.py
import os
from collections import Counter
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import joblib

# ================================
# IMPORTS DEL PROYECTO (MANUAL)
# ================================

# 🔁 K-Means jerárquico MANUAL
from src.predecir_manual import predecir   # misma firma que antes

# Bayes (NO cambia)
from src.bayes import (
    estimar_posterior_secuencial,
    estimar_posterior_por_conteo,
)

# Audio: MISMO pipeline
from src.features_audio import extract_features_from_file
from src.preprocess_audio import preprocess_audio_file


# ================================
# CONFIGURACIÓN
# ================================

# Carpeta con las 10 imágenes
CARPETA_MUESTRA = "data/muestras"

# Audio
INPUT_WAV        = "data_voz/pruebas/comando.wav"
TMP_PREPROCESSED = "data_voz/tmp_preprocessed.wav"

# 🔁 MODELOS MANUALES DE VOZ
MODEL_PATH         = "data_voz/models_manual/knn_model.pkl"
SCALER_PATH        = "data_voz/models_manual/scaler.pkl"
FEATURE_NAMES_PATH = "data_voz/models_manual/feature_names.pkl"

SAMPLE_RATE = 16000
N_MFCC      = 13


# ================================
# FUNCIONES AUXILIARES
# ================================

def clasificar_muestra(carpeta_muestra):
    """
    Clasifica todas las imágenes de la muestra usando
    el K-Means jerárquico MANUAL.
    """
    piezas = []

    if not os.path.isdir(carpeta_muestra):
        print(f"[ERROR] La carpeta de muestra no existe: {carpeta_muestra}")
        return piezas

    archivos = sorted(os.listdir(carpeta_muestra))
    if not archivos:
        print(f"[ADVERTENCIA] La carpeta {carpeta_muestra} está vacía.")
        return piezas

    for nombre_archivo in archivos:
        if not nombre_archivo.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        ruta_img = os.path.join(carpeta_muestra, nombre_archivo)
        try:
            clase, info = predecir(ruta_img)
            piezas.append(clase)
            print(f"[VISION] {nombre_archivo} -> {clase}")
        except Exception as e:
            print(f"[ERROR] Clasificando {nombre_archivo}: {e}")

    return piezas


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
    Predicción de comando de voz usando KNN MANUAL.
    Devuelve (comando, proba_dict)
    """

    if not os.path.exists(input_wav):
        raise FileNotFoundError(f"Archivo de audio no encontrado: {input_wav}")

    preprocess_audio_file(
        input_path=input_wav,
        output_path=tmp_preprocessed,
        sample_rate=sample_rate,
    )

    feats = extract_features_from_file(
        tmp_preprocessed,
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
    )

    feature_names = joblib.load(feature_names_path)
    x_vec = np.array([feats[name] for name in feature_names], dtype=float).reshape(1, -1)

    knn = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    x_scaled = scaler.transform(x_vec)
    predicted_label = knn.predict(x_scaled)[0]

    if hasattr(knn, "predict_proba"):
        proba = knn.predict_proba(x_scaled)[0]
        proba_dict = {cls: float(p) for cls, p in zip(knn.classes_, proba)}
    else:
        proba_dict = None

    print(f"\n[AUDIO] Predicción de comando: {predicted_label}")
    if proba_dict:
        for cls, p in proba_dict.items():
            print(f"  {cls}: {p:.4f}")

    return predicted_label, proba_dict


def escuchar_comando():
    try:
        comando, _ = predecir_comando_desde_wav(INPUT_WAV)
        return str(comando).strip().lower()
    except Exception as e:
        print(f"[ERROR] Al predecir el comando de voz: {e}")
        return None


def mostrar_conteo_piezas(piezas):
    conteo = Counter(piezas)
    print("\n=== CONTEO DE PIEZAS ===")
    for tipo, cantidad in conteo.items():
        print(f"- {tipo}: {cantidad}")
    print("=======================\n")


def mostrar_estimacion_bayes(piezas):
    posterior_final, _ = estimar_posterior_secuencial(piezas)
    caja_mas_prob = max(posterior_final, key=posterior_final.get)

    print("\n=== ESTIMACIÓN BAYESIANA ===")
    for caja, p in posterior_final.items():
        print(f"- Caja {caja}: {p:.4f}")
    print(f"\nCaja más probable: {caja_mas_prob}")
    print("============================\n")


def ejecutar_accion(piezas, comando):
    if comando == "contar":
        mostrar_conteo_piezas(piezas)
        return False

    elif comando == "proporcion":
        mostrar_estimacion_bayes(piezas)
        return False

    elif comando == "salir":
        mostrar_conteo_piezas(piezas)
        mostrar_estimacion_bayes(piezas)
        print("[INFO] Finalizando programa.")
        return True

    else:
        print(f"[WARN] Comando no reconocido: {comando}")
        return False


# ================================
# MAIN
# ================================

def main():
    print("=====================================")
    print("  AGENTE INTELIGENTE (VERSIÓN MANUAL) ")
    print("=====================================\n")

    print(f"[INFO] Clasificando imágenes en {CARPETA_MUESTRA}")
    piezas = clasificar_muestra(CARPETA_MUESTRA)

    if not piezas:
        print("[ERROR] No se clasificaron piezas.")
        return

    print(f"\n[INFO] Total de piezas: {len(piezas)}")
    print(f"[INFO] Etiquetas: {piezas}\n")

    while True:
        print("\nDiga un comando ('contar', 'proporcion', 'salir')")
        input("Audio listo → ENTER ")

        comando = escuchar_comando()
        print(f"[INFO] Comando reconocido: {comando}")

        if ejecutar_accion(piezas, comando):
            break

    print("\n[FIN] Agente finalizado.")


if __name__ == "__main__":
    main()
