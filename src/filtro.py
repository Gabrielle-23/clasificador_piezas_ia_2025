import cv2
import numpy as np


def fresize(image: np.ndarray, size: tuple[int, int] = (252, 560)) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def center_crop(image: np.ndarray,
                frac_h: float ,
                frac_w: float ) -> np.ndarray:
   
    h, w = image.shape[:2]

    ch = int(h * frac_h)  # alto recortado
    cw = int(w * frac_w)  # ancho recortado

    y0 = (h - ch) // 2
    x0 = (w - cw) // 2

    return image[y0:y0 + ch, x0:x0 + cw]

# =========================
#  CONVERSIÓN A ESCALA DE GRISES
# =========================

def fgrayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# =========================
#  FILTROS PASA BAJO
# =========================

kb1 = np.array([[1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]], dtype=np.float32)
kb1 /= kb1.sum()

def fbajok1(image: np.ndarray) -> np.ndarray:
    return cv2.filter2D(image, -1, kb1)


kb2 = np.array([[1, 1, 1],
                [1, 2, 1],
                [1, 1, 1]], dtype=np.float32)
kb2 /= kb2.sum()

def fbajok2(image: np.ndarray) -> np.ndarray:
    return cv2.filter2D(image, -1, kb2)


kb3 = np.array([[1, 2, 1],
                [2, 4, 2],
                [1, 2, 1]], dtype=np.float32)
kb3 /= kb3.sum()

def fbajok3(image: np.ndarray) -> np.ndarray:
    return cv2.filter2D(image, -1, kb3)

# =========================
#  MEDIAN BLUR / GAUSSIANO
# =========================

def fmediana(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Filtro de mediana. Bueno para ruido tipo 'sal y pimienta'
    y conserva mejor los bordes.
    """
    return cv2.medianBlur(image, ksize)


def fgauss(image: np.ndarray,
           ksize: tuple[int, int] = (3, 3),
           sigma: float = 1.0) -> np.ndarray:
    """
    Filtro gaussiano suave.
    """
    return cv2.GaussianBlur(image, ksize, sigma)

# =========================
#  FILTRO PASA ALTO
# =========================

ka1 = np.array([[0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]], dtype=np.float32)
# No normalizo ka1 porque es un filtro de realce

def faltok1(image: np.ndarray) -> np.ndarray:
    return cv2.filter2D(image, -1, ka1)

# =========================
#  BINARIZACIÓN
# =========================

def fbinarize(gray: np.ndarray, invert: bool = True) -> np.ndarray:

    # 0) Asegurar tipo y rango
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray = gray.astype(np.uint8)

    # 1) Suavizado
    smooth = cv2.medianBlur(gray, 3)    #MEDIANBLUR ME DA MEJORES RESULTADOS QUE GAUSSIN

    # 2) Umbral global de Otsu
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

    _, bin_img = cv2.threshold(
        smooth,
        0,
        255,
        thresh_type + cv2.THRESH_OTSU
    )

    return bin_img

#================OPERACIONES MORFOLOGICAS=================================
elemto1=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
elemento2=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
elemento3=cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

def open(imagen):
    openedd=cv2.morphologyEx(imagen,cv2.MORPH_OPEN,elemento3)
    return openedd
def close(imagen):
    closedd=cv2.morphologyEx(imagen,cv2.MORPH_CLOSE,elemento3)
    return closedd

# ==========================================================================
#                               PREPROCESAR COMPLETO
# ==========================================================================

def preprocesar(img_bgr,
                size=(252, 560),
                frac_h=0.75,    #RECORTE MAS HORIZONTAL QUE VERTICAL
                frac_w=0.95,
                invert=True):
    
    gray = fgrayscale(img_bgr)

    gray = fresize(gray, size=size)

    #gray = center_crop(gray, frac_h=frac_h, frac_w=frac_w)

    binaria = fbinarize(gray, invert=invert)


    closed=close(binaria)
    #closed=close(closed)

    return gray, binaria,closed


#======================ELIMINAR FONDO AZUL PARA BINARIZAR==============
def eliminar_fondo_azul(img):
    """
    Toma una imagen BGR y devuelve una imagen binaria donde:
    - El fondo azul queda negro
    - El objeto (tornillo, tuerca, etc.) queda blanco
    """

    # Convertir a HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Rango del azul del fondo (puede ajustarse)
    rango_bajo  = np.array([90, 50, 40])     # H,S,V mínimo
    rango_alto  = np.array([130, 255, 255])  # H,S,V máximo

    # Máscara donde el fondo azul es blanco
    mask_blue = cv2.inRange(hsv, rango_bajo, rango_alto)

    # Invertimos la máscara: fondo negro, objeto blanco
    mask_objeto = cv2.bitwise_not(mask_blue)

    # Limpieza morfológica
    kernel = np.ones((5, 5), np.uint8)
    mask_objeto = cv2.morphologyEx(mask_objeto, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_objeto = cv2.morphologyEx(mask_objeto, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask_objeto

#=============================CONTORNOS===================
def filtrar_contornos(mask):
    """
    Recibe una imagen binaria (0 y 255).
    Devuelve solo los contornos relevantes del objeto.
    """

    h, w = mask.shape[:2]
    area_img = w * h

    # Encontrar contornos
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    contornos_filtrados = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, cw, ch = cv2.boundingRect(cnt)

        # 1) Filtrar contornos demasiado pequeños → ruido
        if area < 0.001 * area_img:
            continue

        # 2) Filtrar contorno gigante que rodea toda la imagen (fondo blanco)
        if cw >= 0.95 * w and ch >= 0.95 * h:
            continue

        # Si pasó ambos filtros, es un contorno válido
        contornos_filtrados.append(cnt)

    return contornos_filtrados, hierarchy