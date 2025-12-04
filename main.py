import cv2
import matplotlib.pyplot as plt

from src.binario import binarizar
from src.utils import procesar_carpeta_imagenes
from src.contornos import filtrar_contornos
from src.features_to_csv import extraer_features_carpeta

#df = extraer_features_carpeta()


from src.predecir import predecir

rutas = [
    "data/pruebas/e.jpg",
    "data/pruebas/b.jpg",
    "data/pruebas/c.jpg",
    "data/pruebas/d.jpg",
]

for r in rutas:
    pred, info = predecir(r)
    print(r, "→", pred, "| rama:", info["rama_predicha_nodo1"])
