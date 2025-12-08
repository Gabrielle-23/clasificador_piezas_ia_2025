# train_knn_voice.py

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def train_knn(
    features_csv: str,
    model_path: str,
    scaler_path: str,
    feature_names_path: str,
    test_size: float = 0.3,
    random_state: int = 42,
    n_neighbors: int = 5
) -> None:
    """
    Entrena un KNN para clasificación de comandos de voz.
    Guarda:
      - modelo KNN (model_path)
      - scaler (scaler_path)
      - lista de columnas de features (feature_names_path)
    """
    # 1) Cargar dataset de features
    df = pd.read_csv(features_csv)

    # 2) Separar features (X) y labels (y)
    #    file_path y label NO son features
    X = df.drop(columns=["file_path", "label"])
    y = df["label"]

    # Guardamos el orden exacto de las columnas de X
    feature_names = list(X.columns)
    print(f"Cantidad de features: {len(feature_names)}")
    print("Primeras columnas:", feature_names[:10])

    # 3) Dividir en train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 4) Escalar (normalizar) las features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5) Definir y entrenar el modelo KNN
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

    # 7) Crear carpetas de destino si no existen
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # 8) Guardar modelo, scaler y feature_names
    joblib.dump(knn, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_names, feature_names_path)

    print(f"[OK] Modelo KNN guardado en: {model_path}")
    print(f"[OK] Scaler guardado en: {scaler_path}")
    print(f"[OK] Orden de columnas guardado en: {feature_names_path}")


if __name__ == "__main__":
    FEATURES_CSV        = "data_voz/features/features_audio.csv"
    MODEL_PATH          = "data_voz/models/knn_model.pkl"
    SCALER_PATH         = "data_voz/models/scaler.pkl"
    FEATURE_NAMES_PATH  = "data_voz/models/feature_names.pkl"

    train_knn(
        features_csv=FEATURES_CSV,
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
        feature_names_path=FEATURE_NAMES_PATH,
        test_size=0.3,
        random_state=42,
        n_neighbors=5,
    )
