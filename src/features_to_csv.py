import os
import cv2
import pandas as pd

from src.features import extraer_features_desde_mask


def inferir_clase_desde_nombre(nombre_archivo: str) -> str:
    """
    Intenta inferir la clase real de la pieza a partir del nombre del archivo.
    Busca palabras clave en minúsculas: 'arandela', 'tuerca', 'tornillo', 'clavo'.
    """
    n = nombre_archivo.lower()
    if "arandela" in n:
        return "arandela"
    if "tuerca" in n:
        return "tuerca"
    if "tornillo" in n:
        return "tornillo"
    if "clavo" in n:
        return "clavo"
    # Si no matchea nada conocido:
    return "desconocida"


def extraer_features_carpeta(
    carpeta_entrada="data/raso/binario",
    ruta_csv_salida="data/features/features.csv",
    extensiones_validas=(".png"),
    min_area_rel=0.001,
    min_hole_area_rel=0.0005,
    epsilon_factor=0.01,
):
    """
    Recorre todas las imágenes binarias de una carpeta, extrae las
    features de cada objeto y genera un DataFrame + un CSV.

    Si se llama sin argumentos: usa las carpetas por defecto.
    """

    filas = []

    # Verificar carpeta de entrada
    if not os.path.isdir(carpeta_entrada):
        raise FileNotFoundError(f"La carpeta de entrada no existe: {carpeta_entrada}")

    # Recorrer los archivos
    for archivo in sorted(os.listdir(carpeta_entrada)):
        if not archivo.lower().endswith(extensiones_validas):
            continue

        ruta_img = os.path.join(carpeta_entrada, archivo)
        print(f"Procesando {ruta_img}")

        mask = cv2.imread(ruta_img, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"⚠ No se pudo leer: {ruta_img}")
            continue

        lista_feats = extraer_features_desde_mask(
            mask,
            min_area_rel=min_area_rel,
            min_hole_area_rel=min_hole_area_rel,
            epsilon_factor=epsilon_factor,
        )

        if not lista_feats:
            print(f"⚠ No se encontraron objetos válidos en {archivo}")
            continue

        clase_real = inferir_clase_desde_nombre(archivo)

        # Cada objeto detectado → una fila
        for obj_id, feats in enumerate(lista_feats):
            fila = {
                "filename": archivo,
                "object_id": obj_id,
                "clase_real": clase_real,  # <-- NUEVA columna
            }
            fila.update(feats)
            filas.append(fila)

    # Crear DataFrame
    df = pd.DataFrame(filas)

    # Guardar CSV
    carpeta_salida = os.path.dirname(ruta_csv_salida)
    if carpeta_salida:
        os.makedirs(carpeta_salida, exist_ok=True)

    df.to_csv(ruta_csv_salida, index=False)
    print(f"\n✔ CSV guardado en: {ruta_csv_salida}")

    return df


if __name__ == "__main__":
    # Llamada por defecto
    df = extraer_features_carpeta()
    print(df.head())
