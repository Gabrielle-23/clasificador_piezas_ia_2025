import os
from collections import Counter
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import joblib

# ==== IMPORTS DE TU PROYECTO ====
# K-Means jerárquico de piezas
from src.predecir import predecir  # devuelve (clase_final, info)

# Bayes: estimador de caja
from src.bayes import (
    estimar_posterior_secuencial,
    estimar_posterior_por_conteo,
)

# Audio: usamos el MISMO pipeline que en tu predecir_audio / predict_knn_voice
from src.features_audio import extract_features_from_file
from src.preprocess_audio import preprocess_audio_file


# ================================
# CONFIGURACIÓN
# ================================

# Carpeta con las 10 imágenes de la muestra
CARPETA_MUESTRA = "data/muestras"

# Audio y modelos del KNN de voz
INPUT_WAV          = "data_voz/pruebas/comando.wav"   # AUDIO RASO ('contar', 'proporcion' o 'salir')
TMP_PREPROCESSED   = "data_voz/tmp_preprocessed.wav"  

# archivo temporal preprocesado

MODEL_PATH         = "data_voz/models/knn_model.pkl"
SCALER_PATH        = "data_voz/models/scaler.pkl"
FEATURE_NAMES_PATH = "data_voz/models/feature_names.pkl"

SAMPLE_RATE        = 16000
N_MFCC             = 13


# ================================
# FUNCIONES AUXILIARES
# ================================

def clasificar_muestra(carpeta_muestra):
    """
    Recorre la carpeta de la muestra, aplica tu K-Means jerárquico
    (función predecir de predecir.py) a cada imagen y devuelve una lista
    con las etiquetas de las piezas.
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
    Versión que **devuelve** la etiqueta del comando.

    Es básicamente el mismo pipeline que en tu script de KNN de voz:
      1) Preprocesar audio crudo -> tmp_preprocessed.wav
      2) Extraer features
      3) Ordenar features según feature_names
      4) Escalar
      5) Predecir con KNN

    Devuelve:
      - predicted_label (str)
      - proba_dict (dict o None)
    """

    if not os.path.exists(input_wav):
        raise FileNotFoundError(f"Archivo de audio no encontrado: {input_wav}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler no encontrado: {scaler_path}")
    if not os.path.exists(feature_names_path):
        raise FileNotFoundError(f"feature_names no encontrado: {feature_names_path}")

    # 1) Preprocesar audio crudo
    preprocess_audio_file(
        input_path=input_wav,
        output_path=tmp_preprocessed,
        sample_rate=sample_rate,
    )

    # 2) Extraer features del audio preprocesado
    feats = extract_features_from_file(
        tmp_preprocessed,
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
    )

    # 3) Cargar el orden de columnas usado en el entrenamiento
    feature_names = joblib.load(feature_names_path)

    missing = [f for f in feature_names if f not in feats]
    extra   = [f for f in feats.keys() if f not in feature_names]

    if missing:
        raise ValueError(
            f"Faltan features en la predicción: {missing}. "
            "Revisá que extract_audio_features.py coincida con el usado en entrenamiento."
        )

    if extra:
        print(f"[WARN] Features extra que no se usan en el modelo: {extra}")

    # 4) Vector de entrada en el mismo orden de columnas
    x_vec = np.array([feats[name] for name in feature_names], dtype=float).reshape(1, -1)

    # 5) Cargar modelo y scaler
    knn = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # 6) Escalar
    x_scaled = scaler.transform(x_vec)

    # 7) Predicción
    predicted_label = knn.predict(x_scaled)[0]

    if hasattr(knn, "predict_proba"):
        proba = knn.predict_proba(x_scaled)[0]
        classes = knn.classes_
        proba_dict = {cls: float(p) for cls, p in zip(classes, proba)}
    else:
        proba_dict = None

    print(f"\n\n[AUDIO] Archivo: {input_wav}")
    print(f"[AUDIO] Predicción de comando: {predicted_label}")
    if proba_dict is not None:
        print("[AUDIO] Probabilidades por clase:")
        for cls, p in proba_dict.items():
            print(f"  {cls}: {p:.4f}")

    return predicted_label, proba_dict


def escuchar_comando():
    """
    En esta versión simple, asumimos que el audio ya fue grabado
    y guardado en INPUT_WAV. Solo lo procesamos.
    """
    try:
        comando, _ = predecir_comando_desde_wav(INPUT_WAV)
        comando = str(comando).strip().lower()
        return comando
    except Exception as e:
        print(f"[ERROR] Al predecir el comando de voz: {e}")
        return None


def mostrar_conteo_piezas(piezas):
    """
    Muestra el conteo de cada tipo de pieza en la muestra.
    """
    if not piezas:
        print("[ADVERTENCIA] No hay piezas para contar.")
        return

    conteo = Counter(piezas)
    print("\n=== CONTEO DE PIEZAS EN LA MUESTRA ===")
    for tipo, cantidad in conteo.items():
        print(f"- {tipo}: {cantidad}")
    print("======================================\n")


def mostrar_estimacion_bayes(piezas):
    """
    Usa tu estimador bayesiano para calcular P(caja | muestra)
    y muestra las probabilidades de cada caja.
    """
    if not piezas:
        print("[ADVERTENCIA] No hay piezas en la muestra para estimar la caja.")
        return

    try:
        posterior_final, historial = estimar_posterior_secuencial(piezas)
        # Alternativamente podrías usar estimar_posterior_por_conteo(piezas)
        # posterior_final = estimar_posterior_por_conteo(piezas)
    except Exception as e:
        print(f"[ERROR] Al calcular el posterior bayesiano: {e}")
        return

    caja_mas_prob = max(posterior_final, key=posterior_final.get)

    print("\n=== ESTIMACIÓN BAYESIANA DE LA CAJA DE ORIGEN ===")
    for caja, p in posterior_final.items():
        print(f"- Caja {caja}: {p:.4f}")
    print(f"\nCaja más probable: {caja_mas_prob}")
    print("=================================================\n")


def ejecutar_accion(piezas, comando):
    """
    Ejecuta la acción pedida por el comando de voz:
      - 'contar'     -> mostrar conteo de piezas
      - 'proporcion' -> estimar caja de origen con Bayes
      - 'salir'      -> mostrar resumen y terminar
    Devuelve True si hay que salir del programa, False en caso contrario.
    """
    if comando is None:
        print("[ADVERTENCIA] No se obtuvo un comando válido.")
        return False

    if comando == "contar":
        mostrar_conteo_piezas(piezas)
        return False

    elif comando == "proporcion":
        mostrar_estimacion_bayes(piezas)
        return False

    elif comando == "salir":
        print("\n[INFO] Comando 'salir' recibido. Mostrando resumen final...\n")
        mostrar_conteo_piezas(piezas)
        mostrar_estimacion_bayes(piezas)
        print("[INFO] Finalizando el programa.")
        return True

    else:
        print(f"[ADVERTENCIA] Comando no reconocido: '{comando}'")
        print("Se esperaban: 'contar', 'proporcion' o 'salir'.")
        return False


# ================================
# MAIN PRINCIPAL
# ================================

def main():
    print("=====================================")
    print("  AGENTE INTELIGENTE - PROYECTO 2025 ")
    print("=====================================\n")

    # 1) Clasificar las 10 piezas de la muestra
    print(f"[INFO] Clasificando imágenes en: {CARPETA_MUESTRA}")
    piezas = clasificar_muestra(CARPETA_MUESTRA)

    if not piezas:
        print("[ERROR] No se obtuvieron piezas de la muestra. Saliendo.")
        return

    print(f"\n[INFO] Muestra clasificada. Total de piezas: {len(piezas)}")
    print(f"[INFO] Etiquetas obtenidas: {piezas}\n")

    # 2) Loop de comandos de voz
    #    Podés dejar que el profe dé varios comandos uno tras otro,
    #    hasta decir 'salir', o cortar después del primero.
    while True:
        print("\nDiga un comando de voz ('contar', 'proporcion' o 'salir').")
        print(f"[INFO] Guardar audio en: {INPUT_WAV}")
        input("Audio listo --> presione ENTER... ")

        comando = escuchar_comando()
        print(f"\n[INFO] Comando reconocido -->> \"{comando}\"")

        terminar = ejecutar_accion(piezas, comando)
        if terminar:
            break

        # Si querés que el sistema sólo acepte UN comando y termine,
        # descomentá esta línea:
        # break

    print("\n[FIN] Programa finalizado.")


if __name__ == "__main__":
    main()
