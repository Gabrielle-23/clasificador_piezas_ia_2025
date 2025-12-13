# src/predecir_manual.py
from pathlib import Path
import json

import cv2
import numpy as np
import joblib

from src.binario import binarizar
from src.features import extraer_features_desde_mask

# ✅ modelos manuales (NO pisa los modelos sklearn)
MODELS_DIR = Path("data/models_manual")

# Parámetros de features usados en el CSV (ver features_to_csv.py)
MIN_AREA_REL = 0.001
MIN_HOLE_AREA_REL = 0.0005
EPSILON_FACTOR = 0.01


def _cargar_info_json(path_json: Path):
    with open(path_json, "r", encoding="utf-8") as f:
        return json.load(f)


def _lookup_mapping(mapping: dict, key_int: int):
    """
    mapping viene del JSON, con claves string.
    Este helper intenta mapping[str(key)] y luego mapping[key].
    """
    key_str = str(key_int)
    if key_str in mapping:
        return mapping[key_str]
    if key_int in mapping:
        return mapping[key_int]
    raise KeyError(f"Clave {key_int} no encontrada en mapping {mapping}.")


def _extraer_features_desde_imagen(ruta_imagen: Path) -> dict:
    """
    Dada la ruta de una imagen:
    - si es color, se binariza con binarizar(),
    - si es de 1 canal, se toma directamente como máscara,
    luego se llama a extraer_features_desde_mask con los mismos
    parámetros que se usaron para generar features.csv.
    Devuelve el diccionario de features del objeto más grande.
    """
    img = cv2.imread(str(ruta_imagen), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {ruta_imagen}")

    # Si la imagen es 2D o tiene 1 canal -> asumimos que ya es máscara binaria
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        mask = img
    else:
        # Imagen color BGR con fondo verde -> usar tu binarización estándar
        mask = binarizar(img)  # devuelve máscara 0/255 de tamaño fijo

    lista_feats = extraer_features_desde_mask(
        mask,
        min_area_rel=MIN_AREA_REL,
        min_hole_area_rel=MIN_HOLE_AREA_REL,
        epsilon_factor=EPSILON_FACTOR,
    )

    if not lista_feats:
        raise ValueError(
            f"No se encontraron objetos válidos en la imagen binaria: {ruta_imagen}"
        )

    # Si hubiera más de un objeto, usamos el de mayor área (como criterio razonable)
    if len(lista_feats) > 1:
        lista_feats = sorted(
            lista_feats,
            key=lambda d: d.get("area", 0.0),
            reverse=True,
        )

    return lista_feats[0]


def predecir(ruta_imagen):
    """
    Clasifica una imagen en {arandela, tuerca, clavo, tornillo}
    usando el sistema jerárquico entrenado (manual).

    Parámetros
    ----------
    ruta_imagen : str o Path

    Returns
    -------
    clase_final : str
        Clase predicha.
    info : dict
        Información adicional sobre la predicción.
    """
    ruta_imagen = Path(ruta_imagen)

    if not ruta_imagen.is_file():
        raise FileNotFoundError(f"No se encontró la imagen: {ruta_imagen}")

    # 1) Extraer features con el MISMO pipeline del entrenamiento
    feats = _extraer_features_desde_imagen(ruta_imagen)

    # 2) Cargar modelos y metadatos de cada nodo (manuales)

    # Nodo 1: aro vs palo
    scaler1 = joblib.load(MODELS_DIR / "nodo1_aro_palo_scaler.joblib")
    kmeans1 = joblib.load(MODELS_DIR / "nodo1_aro_palo_kmeans.joblib")
    info1 = _cargar_info_json(MODELS_DIR / "nodo1_aro_palo_info.json")
    features_n1 = info1["features"]
    mapping_branch = info1["cluster_to_branch"]

    # Nodo 2: arandela vs tuerca
    scaler2 = joblib.load(MODELS_DIR / "nodo2_arandela_tuerca_scaler.joblib")
    kmeans2 = joblib.load(MODELS_DIR / "nodo2_arandela_tuerca_kmeans.joblib")
    info2 = _cargar_info_json(MODELS_DIR / "nodo2_arandela_tuerca_info.json")
    features_n2 = info2["features"]
    mapping_at = info2["cluster_to_class"]

    # Nodo 3: clavo vs tornillo
    scaler3 = joblib.load(MODELS_DIR / "nodo3_clavo_tornillo_scaler.joblib")
    kmeans3 = joblib.load(MODELS_DIR / "nodo3_clavo_tornillo_kmeans.joblib")
    info3 = _cargar_info_json(MODELS_DIR / "nodo3_clavo_tornillo_info.json")
    features_n3 = info3["features"]
    mapping_ct = info3["cluster_to_class"]

    # 3) Predicción jerárquica

    # Nodo 1: aro vs palo
    x1 = np.array([[feats[f] for f in features_n1]], dtype=np.float64)
    x1_scaled = scaler1.transform(x1)
    cl1 = int(kmeans1.predict(x1_scaled)[0])
    branch_pred = _lookup_mapping(mapping_branch, cl1)  # 'aro' o 'palo'

    if branch_pred == "aro":
        # Nodo 2: arandela vs tuerca
        x2 = np.array([[feats[f] for f in features_n2]], dtype=np.float64)
        x2_scaled = scaler2.transform(x2)
        cl2 = int(kmeans2.predict(x2_scaled)[0])
        clase_final = _lookup_mapping(mapping_at, cl2)
        nodo_final = 2
        cluster_final = cl2
    else:
        # Nodo 3: clavo vs tornillo
        x3 = np.array([[feats[f] for f in features_n3]], dtype=np.float64)
        x3_scaled = scaler3.transform(x3)
        cl3 = int(kmeans3.predict(x3_scaled)[0])
        clase_final = _lookup_mapping(mapping_ct, cl3)
        nodo_final = 3
        cluster_final = cl3

    info = {
        "ruta_imagen": str(ruta_imagen),
        "rama_predicha_nodo1": branch_pred,
        "cluster_nodo1": cl1,
        "nodo_final": nodo_final,
        "cluster_nodo_final": cluster_final,
        "clase_predicha": clase_final,
        "features_usados": feats,
    }

    return clase_final, info


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python -m src.predecir_manual RUTA_IMAGEN")
        sys.exit(1)

    ruta = sys.argv[1]
    pred, info = predecir(ruta)
    print(f"Predicción para {ruta}: {pred}")
