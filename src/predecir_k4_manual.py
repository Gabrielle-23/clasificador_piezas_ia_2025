"""
Predicción con KMeans k=4 MANUAL (sin sklearn) usando TU pipeline de preprocesamiento de imagen.

- Reutiliza binarizar() y extraer_features_desde_mask() tal como en src/predecir_manual.py
- Carga el modelo manual guardado por train_kmeans_k4_manual.py
- NO pisa tu sistema jerárquico actual.

Uso:
  python predecir_k4_manual.py RUTA_IMAGEN
"""

from pathlib import Path
import json

import cv2
import numpy as np
import joblib

from src.binario import binarizar
from src.features import extraer_features_desde_mask

# Carpeta NUEVA para k=4
MODELS_DIR = Path("data/models_k4_manual")

# mismos parámetros que usaste para generar features.csv
MIN_AREA_REL = 0.001
MIN_HOLE_AREA_REL = 0.0005
EPSILON_FACTOR = 0.01


def _cargar_info_json(path_json: Path):
    with open(path_json, "r", encoding="utf-8") as f:
        return json.load(f)


def _lookup_mapping(mapping: dict, key_int: int):
    key_str = str(key_int)
    if key_str in mapping:
        return mapping[key_str]
    if key_int in mapping:
        return mapping[key_int]
    raise KeyError(f"Clave {key_int} no encontrada en mapping {mapping}.")


def _extraer_features_desde_imagen(ruta_imagen: Path) -> dict:
    img = cv2.imread(str(ruta_imagen), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {ruta_imagen}")

    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        mask = img
    else:
        mask = binarizar(img)

    lista_feats = extraer_features_desde_mask(
        mask,
        min_area_rel=MIN_AREA_REL,
        min_hole_area_rel=MIN_HOLE_AREA_REL,
        epsilon_factor=EPSILON_FACTOR,
    )

    if not lista_feats:
        raise ValueError(f"No se encontraron objetos válidos en la imagen: {ruta_imagen}")

    if len(lista_feats) > 1:
        lista_feats = sorted(lista_feats, key=lambda d: d.get("area", 0.0), reverse=True)

    return lista_feats[0]


def predecir_k4(ruta_imagen):
    ruta_imagen = Path(ruta_imagen)
    if not ruta_imagen.is_file():
        raise FileNotFoundError(f"No se encontró la imagen: {ruta_imagen}")

    feats = _extraer_features_desde_imagen(ruta_imagen)

    scaler = joblib.load(MODELS_DIR / "k4_scaler.joblib")
    kmeans = joblib.load(MODELS_DIR / "k4_kmeans.joblib")
    info = _cargar_info_json(MODELS_DIR / "k4_info.json")

    features = info["features"]
    mapping = info["cluster_to_class"]

    x = np.array([[feats[f] for f in features]], dtype=np.float64)
    xs = scaler.transform(x)
    cl = int(kmeans.predict(xs)[0])

    clase = _lookup_mapping(mapping, cl)

    out = {
        "ruta_imagen": str(ruta_imagen),
        "cluster_predicho": cl,
        "clase_predicha": clase,
        "features_usadas": features,
        "models_dir": str(MODELS_DIR),
    }
    return clase, out


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python predecir_k4_manual.py RUTA_IMAGEN")
        sys.exit(1)

    pred, info = predecir_k4(sys.argv[1])
    print(f"Predicción para {sys.argv[1]}: {pred}")
    # si querés ver detalles:
    # print(info)
