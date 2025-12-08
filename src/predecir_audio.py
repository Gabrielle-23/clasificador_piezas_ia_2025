# predict_knn_voice.py

import os
import numpy as np
import joblib

from src.features_audio import extract_features_from_file
from src.preprocess_audio import preprocess_audio_file


def predict_command(
    input_wav: str,
    model_path: str,
    scaler_path: str,
    feature_names_path: str,
    tmp_preprocessed: str = "tmp_preprocessed.wav",
    sample_rate: int = 16000,
    n_mfcc: int = 13
) -> None:
    """
    Predice el comando hablado en input_wav usando el modelo KNN entrenado.
    Pipeline:
      1) Preprocesar audio crudo -> tmp_preprocessed
      2) Extraer features (dict)
      3) Ordenar features según feature_names guardado en entrenamiento
      4) Escalar con scaler
      5) Predecir con modelo KNN
    """

    # 1) Chequear que existan modelo, scaler y feature_names
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler no encontrado: {scaler_path}")
    if not os.path.exists(feature_names_path):
        raise FileNotFoundError(f"feature_names no encontrado: {feature_names_path}")

    # 2) Preprocesar audio crudo -> tmp_preprocessed.wav
    preprocess_audio_file(
        input_path=input_wav,
        output_path=tmp_preprocessed,
        sample_rate=sample_rate
    )

    # 3) Extraer features del audio preprocesado
    feats = extract_features_from_file(
        tmp_preprocessed,
        sample_rate=sample_rate,
        n_mfcc=n_mfcc
    )

    # 4) Cargar el orden de columnas usado en el entrenamiento
    feature_names = joblib.load(feature_names_path)

    # Validación rápida: chequear que no falte ninguna feature necesaria
    missing = [f for f in feature_names if f not in feats]
    extra   = [f for f in feats.keys() if f not in feature_names]

    if missing:
        raise ValueError(
            f"Faltan features en la predicción: {missing}. "
            f"Revisa que extract_audio_features.py coincida con el usado en entrenamiento."
        )

    # (Las extra no rompen, pero las avisamos por consola)
    if extra:
        print(f"[WARN] Features extra que no se usan en el modelo: {extra}")

    # 5) Construir el vector de entrada en el mismo orden de columnas
    x_vec = np.array([feats[name] for name in feature_names], dtype=float).reshape(1, -1)

    # 6) Cargar modelo y scaler
    knn = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # 7) Escalar
    x_scaled = scaler.transform(x_vec)

    # 8) Predicción
    predicted_label = knn.predict(x_scaled)[0]

    # Si el modelo fue entrenado con probas, podemos mostrar predict_proba
    if hasattr(knn, "predict_proba"):
        proba = knn.predict_proba(x_scaled)[0]
        classes = knn.classes_
        proba_dict = {cls: p for cls, p in zip(classes, proba)}
    else:
        proba_dict = None

    print(f"Archivo: {input_wav}")
    print(f"Predicción: {predicted_label}")

    if proba_dict is not None:
        print("Probabilidades por clase:")
        for cls, p in proba_dict.items():
            print(f"  {cls}: {p:.4f}")


if __name__ == "__main__":
    INPUT_WAV          = "data_voz/pruebas/s.wav"  # cámbialo por el audio que quieras probar
    MODEL_PATH         = "data_voz/models/knn_model.pkl"
    SCALER_PATH        = "data_voz/models/scaler.pkl"
    FEATURE_NAMES_PATH = "data_voz/models/feature_names.pkl"

    predict_command(
        input_wav=INPUT_WAV,
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
        feature_names_path=FEATURE_NAMES_PATH,
        tmp_preprocessed="tmp_preprocessed.wav",
        sample_rate=16000,
        n_mfcc=13
    )
