import os
import cv2
from src.binario import binarizar

def procesar_carpeta_imagenes(
        carpeta_entrada="data/raso",
        carpeta_salida="data/raso/binario"
    ):
    """
    Recorre todas las imágenes de carpeta_entrada,
    aplica binarizar() y guarda la máscara como PNG sin pérdida
    en carpeta_salida, manteniendo el mismo nombre base.
    """

    os.makedirs(carpeta_salida, exist_ok=True)

    extensiones_validas = (".jpg", ".jpeg", ".png")

    for archivo in os.listdir(carpeta_entrada):

        if archivo.lower().endswith(extensiones_validas):

            ruta_in = os.path.join(carpeta_entrada, archivo)

            # Cambiar SIEMPRE a extensión .png (sin compresión con pérdida)
            nombre_base, _ = os.path.splitext(archivo)
            archivo_out = nombre_base + ".png"
            ruta_out = os.path.join(carpeta_salida, archivo_out)

            print(f"Procesando: {ruta_in} -> {ruta_out}")

            img = cv2.imread(ruta_in)
            if img is None:
                print(f"⚠ No se pudo leer {ruta_in}")
                continue

            # Binarizar imagen original
            mask = binarizar(img)

            # Guardar como PNG (formato sin pérdida)
            cv2.imwrite(ruta_out, mask)

    print("\n✔ Finalizado: máscaras binarias guardadas en formato PNG")



















'''
def procesar_carpeta_imagenes(
        carpeta_entrada="data/raso",
        carpeta_salida="data/raso/binario"
    ):
    """
    Recorre todas las imágenes de data/raw/images,
    les aplica eliminar_fondo_azul y guarda la máscara
    en data/intermed manteniendo el mismo nombre.
    """

    # Crear carpeta de salida si no existe
    os.makedirs(carpeta_salida, exist_ok=True)

    # Archivos válidos
    extensiones_validas = (".jpg", ".jpeg", ".png")

    # Recorrer carpeta
    for archivo in os.listdir(carpeta_entrada):
        if archivo.lower().endswith(extensiones_validas):

            ruta_in = os.path.join(carpeta_entrada, archivo)
            ruta_out = os.path.join(carpeta_salida, archivo)

            print(f"Procesando: {ruta_in}")

            # 1) Leer imagen
            img = cv2.imread(ruta_in)
            if img is None:
                print(f"⚠ No se pudo leer {ruta_in}")
                continue


            # 3) Aplicar binarización / eliminación de fondo azul
            mask = binarizar(img)

            # 4) Guardar máscara binarizada
            cv2.imwrite(ruta_out, mask)

    print("\n✔ Finalizado: imágenes guardadas en", carpeta_salida)
'''