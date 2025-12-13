# src/_knn_manual.py
"""
Versión MANUAL (sin sklearn) del trainer KNN de voz.
Replica el comportamiento de knn.py (train_knn_voice.py), pero:
- Usa split/scaler/knn/metrics de src.ml_manual
- Guarda en data_voz/models_manual/ (no pisa data_voz/models/)
"""

import os
import joblib
import pandas as pd

from src.ml_manual.split import train_test_split
from src.ml_manual.scaler import StandardScaler
from src.ml_manual.knn import KNeighborsClassifier
from src.ml_manual.metrics import accuracy_score, confusion_matrix, classification_report


def train_knn(
    features_csv: str,
    model_path: str,
    scaler_path: str,
    feature_names_path: str,
    test_size: float = 0.3,
    random_state: int = 42,
    n_neighbors: int = 5
) -> None:
    # 1) Cargar dataset de features
    df = pd.read_csv(features_csv)

    # 2) Separar features (X) y labels (y)
    X = df.drop(columns=["file_path", "label"])
    y = df["label"].values

    feature_names = list(X.columns)
    print(f"Cantidad de features: {len(feature_names)}")
    print("Primeras columnas:", feature_names[:10])

    # 3) Dividir en train / test (estratificado como tu versión sklearn) :contentReference[oaicite:6]{index=6}
    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 4) Escalar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5) Entrenar KNN (weights="distance", euclidean como tu versión) :contentReference[oaicite:7]{index=7}
    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights="distance",
        metric="euclidean"
    )
    knn.fit(X_train_scaled, y_train)

    # 6) Evaluar
    y_pred = knn.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)
    cr  = classification_report(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print("Matriz de confusión:")
    print(cm)
    print("Reporte de clasificación:")
    print(cr)

    # 7) Guardar (separado)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump(knn, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_names, feature_names_path)

    print(f"[OK] Modelo KNN guardado en: {model_path}")
    print(f"[OK] Scaler guardado en: {scaler_path}")
    print(f"[OK] Orden de columnas guardado en: {feature_names_path}")


if __name__ == "__main__":
    FEATURES_CSV        = "data_voz/features/features_audio.csv"
    MODEL_PATH          = "data_voz/models_manual/knn_model.pkl"
    SCALER_PATH         = "data_voz/models_manual/scaler.pkl"
    FEATURE_NAMES_PATH  = "data_voz/models_manual/feature_names.pkl"

    train_knn(
        features_csv=FEATURES_CSV,
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
        feature_names_path=FEATURE_NAMES_PATH,
        test_size=0.3,
        random_state=42,
        n_neighbors=5,
    )
