import cv2
import numpy as np

def filtrar_contornos(
    mask,
    min_area_rel=0.001,
    borde_rel=0.95,
    mode=cv2.RETR_TREE
):
    """
    Recibe una imagen binaria (0 y 255) y devuelve solo los contornos relevantes.

    Parámetros
    ----------
    mask : np.ndarray
        Máscara binaria (una o tres canales). Idealmente 0/255.
    min_area_rel : float
        Área mínima relativa al área total de la imagen para aceptar un contorno.
        Sirve para eliminar ruido chico. Ejemplo: 0.001 -> 0.1% del área.
    borde_rel : float
        Umbral relativo del bounding box para descartar el contorno que,
        en la práctica, rodea casi toda la imagen (marco/fondo).
    mode : int
        Modo de recuperación de contornos de OpenCV.
        - cv2.RETR_TREE (por defecto) -> conserva jerarquía interna (sirve
          para tuercas / arandelas con agujero).
        - cv2.RETR_EXTERNAL -> solo contornos externos (si alguna vez
          quisieras ignorar agujeros internos).

    Devuelve
    --------
    contornos_filtrados : list[np.ndarray]
        Lista de contornos aceptados, ordenados de mayor a menor área.
    hierarchy : np.ndarray o None
        Jerarquía que devuelve cv2.findContours.
    """

    if mask is None:
        raise ValueError("mask es None. Revisá el flujo de binarización.")

    # Aseguramos que sea imagen de un canal
    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask

    h, w = mask_gray.shape[:2]
    area_img = float(w * h)

    # Encontrar contornos
    contours, hierarchy = cv2.findContours(
        mask_gray, mode, cv2.CHAIN_APPROX_SIMPLE
    )

    contornos_filtrados = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, cw, ch = cv2.boundingRect(cnt)

        # 1) Filtrar contornos demasiado pequeños → ruido
        if area < min_area_rel * area_img:
            continue

        # 2) Filtrar contorno gigante que rodea casi toda la imagen (fondo / marco)
        if cw >= borde_rel * w and ch >= borde_rel * h:
            continue

        contornos_filtrados.append(cnt)

    # Opcional pero útil: ordenarlos por área (el más grande primero)
    contornos_filtrados.sort(key=cv2.contourArea, reverse=True)

    return contornos_filtrados, hierarchy
