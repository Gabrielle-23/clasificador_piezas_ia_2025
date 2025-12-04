import cv2
import numpy as np

LOW_GREEN  = (35, 40, 40)
HIGH_GREEN = (85, 255, 255)
KERNEL_SIZE = 3

# Tamaño único para tu dataset
OUTPUT_SIZE = (512, 512)   # (ancho, alto)


def binarizar(img):
    """
    Recibe una imagen (BGR) y devuelve una imagen binaria normalizada
    de tamaño fijo OUTPUT_SIZE.
    """

    if img is None:
        raise ValueError("La imagen recibida es None")

    # --- Procesamiento del fondo ---
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

    lower = np.array(LOW_GREEN, dtype=np.uint8)
    upper = np.array(HIGH_GREEN, dtype=np.uint8)
    mask_fondo = cv2.inRange(img_hsv, lower, upper)
    mask_objeto = cv2.bitwise_not(mask_fondo)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))
    mask_limpia = cv2.morphologyEx(mask_objeto, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_limpia = cv2.morphologyEx(mask_limpia, cv2.MORPH_CLOSE, kernel, iterations=1)

    _, binaria = cv2.threshold(mask_limpia, 127, 255, cv2.THRESH_BINARY)

    # --- Normalización del tamaño ---
    binaria_resized = cv2.resize(binaria, OUTPUT_SIZE, interpolation=cv2.INTER_NEAREST)

    return binaria_resized
