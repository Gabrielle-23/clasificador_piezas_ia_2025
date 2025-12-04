import cv2
import numpy as np


def momentos_hu(contour, n=6):
    """
    Calcula los momentos invariantes de Hu para un contorno
    y los devuelve en escala logarítmica.
    """
    m = cv2.moments(contour)
    hu = cv2.HuMoments(m).flatten()

    # Escala logarítmica para evitar valores extremos (manteniendo signo)
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    return hu_log[:n]


def _indices_hijos(hierarchy, idx_padre):
    """
    Devuelve los índices de los contornos hijos de un contorno externo
    según la jerarquía de cv2.findContours (modo RETR_TREE).
    """
    if hierarchy is None:
        return []

    hijos = []
    # hierarchy tiene forma (1, N, 4)
    for j, h in enumerate(hierarchy[0]):
        padre = h[3]
        if padre == idx_padre:
            hijos.append(j)
    return hijos


def extraer_features_objeto(
    contours,
    hierarchy,
    idx_contorno,
    image_shape,
    min_hole_area_rel=0.0005,
    epsilon_factor=0.01,  # <-- NUEVO parámetro
):
    """
    Extrae todas las características de UN objeto, definido por:
    - un contorno externo (idx_contorno) y
    - sus agujeros internos (hijos en la jerarquía).
    """
    cnt = contours[idx_contorno]

    # --- Info de imagen para normalizar ---
    if len(image_shape) == 3:
        h_img, w_img, _ = image_shape
    else:
        h_img, w_img = image_shape
    area_img = float(w_img * h_img)
    escala_longitud = np.sqrt(area_img) if area_img > 0 else 1.0

    # --- Área y perímetro (no normalizados) ---
    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))

    # --- Normalizados ---
    area_norm = area / area_img if area_img > 0 else 0.0
    perimeter_norm = perimeter / escala_longitud if escala_longitud > 0 else 0.0

    # --- Aspect ratio (minAreaRect) ---
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (w_rect, h_rect), angle = rect
    if min(w_rect, h_rect) > 0:
        aspect_ratio = max(w_rect, h_rect) / min(w_rect, h_rect)
    else:
        aspect_ratio = 0.0

    # --- Bounding box axis-aligned (para extent) ---
    x, y, w_box, h_box = cv2.boundingRect(cnt)
    box_area = float(w_box * h_box)
    extent = area / box_area if box_area > 0 else 0.0

    # --- Convex hull / solidity ---
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 0.0

    # --- Circularidad ---
    if perimeter > 0:
        circularity = 4.0 * np.pi * area / (perimeter ** 2)
    else:
        circularity = 0.0

    # --- Aproximación poligonal (Ramer–Douglas–Peucker) ---
    if perimeter > 0:
        epsilon = epsilon_factor * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        num_vertices_rdp = int(len(approx))
        vertices_perimeter_ratio = float(num_vertices_rdp / perimeter)
    else:
        num_vertices_rdp = 0
        vertices_perimeter_ratio = 0.0

    # --- Agujeros internos (hijos en jerarquía) ---
    idx_hijos = _indices_hijos(hierarchy, idx_contorno)

    if len(image_shape) == 3:
        h_img, w_img, _ = image_shape
    else:
        h_img, w_img = image_shape
    area_img = float(w_img * h_img)
    min_hole_area = min_hole_area_rel * area_img

    num_holes = 0
    area_holes_total = 0.0

    for j in idx_hijos:
        # OJO: hierarchy y contours deben tener la misma longitud
        if j < 0 or j >= len(contours):
            continue
        cnt_hijo = contours[j]
        area_hijo = float(cv2.contourArea(cnt_hijo))
        if area_hijo >= min_hole_area:
            num_holes += 1
            area_holes_total += area_hijo

    hole_area_ratio = (area_holes_total / area) if area > 0 else 0.0

    # --- Momentos de Hu (1..6) ---
    hu = momentos_hu(cnt, n=6)

    # Construimos el diccionario de salida
    features = {
        "contour_index": int(idx_contorno),
        "area": area,
        "perimeter": perimeter,
        "area_norm": area_norm,
        "perimeter_norm": perimeter_norm,
        "aspect_ratio": float(aspect_ratio),
        "extent": float(extent),
        "solidity": float(solidity),
        "circularity": float(circularity),
        "num_holes": int(num_holes),
        "hole_area_ratio": float(hole_area_ratio),
        "num_vertices_rdp": num_vertices_rdp,                # <-- NUEVO
        "vertices_perimeter_ratio": vertices_perimeter_ratio, # <-- NUEVO
        "hu1": float(hu[0]),
        "hu2": float(hu[1]),
        "hu3": float(hu[2]),
        "hu4": float(hu[3]),
        "hu5": float(hu[4]),
        "hu6": float(hu[5]),
    }

    return features


def extraer_features_todos(
    contours,
    hierarchy,
    image_shape,
    min_area_rel=0.001,
    min_hole_area_rel=0.0005,
    epsilon_factor=0.01,  # <-- NUEVO
):
    """
    Extrae features para TODOS los objetos de la imagen.
    """
    if hierarchy is None:
        raise ValueError(
            "hierarchy es None. Asegurate de usar cv2.RETR_TREE en findContours."
        )

    if len(hierarchy.shape) != 3 or hierarchy.shape[1] != len(contours):
        raise ValueError(
            f"Dimensiones incompatibles: len(contours)={len(contours)} "
            f"vs hierarchy.shape={hierarchy.shape}. "
            "Ambos deben provenir del mismo findContours (sin filtrar)."
        )

    if len(image_shape) == 3:
        h_img, w_img, _ = image_shape
    else:
        h_img, w_img = image_shape
    area_img = float(w_img * h_img)
    min_area_abs = min_area_rel * area_img

    lista_features = []

    for idx, cnt in enumerate(contours):
        # Sólo contornos externos (padre = -1)
        padre = hierarchy[0][idx][3]
        if padre != -1:
            continue

        area = float(cv2.contourArea(cnt))
        if area < min_area_abs:
            # demasiado pequeño → ruido
            continue

        feats = extraer_features_objeto(
            contours=contours,
            hierarchy=hierarchy,
            idx_contorno=idx,
            image_shape=image_shape,
            min_hole_area_rel=min_hole_area_rel,
            epsilon_factor=epsilon_factor,  # <-- pasamos el valor
        )
        lista_features.append(feats)

    return lista_features


def extraer_features_desde_mask(
    mask,
    min_area_rel=0.001,
    min_hole_area_rel=0.0005,
    epsilon_factor=0.01,  # <-- NUEVO
):
    """
    Helper conveniente: recibe directamente una MASK binaria y devuelve
    las features de todos los objetos usando RETR_TREE.
    """
    # Si la máscara viene en BGR, la pasamos a gris
    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask

    contours, hierarchy = cv2.findContours(
        mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    return extraer_features_todos(
        contours=contours,
        hierarchy=hierarchy,
        image_shape=mask_gray.shape,
        min_area_rel=min_area_rel,
        min_hole_area_rel=min_hole_area_rel,
        epsilon_factor=epsilon_factor,  # <-- idem
    )