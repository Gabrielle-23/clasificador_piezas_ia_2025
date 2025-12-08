# features_audio.py  (antes extract_audio_features.py)

import os
import re  # <-- nuevo
import numpy as np
import pandas as pd
import librosa


def extract_features_from_file(
    file_path: str,
    sample_rate: int = 16000,
    n_mfcc: int = 13
) -> dict:
    """
    Extrae características de un solo archivo de audio:
    - MFCC mean + std
    - ZCR mean + std
    - RMS mean + std
    - Spectral Centroid mean + std
    - Spectral Bandwidth mean + std
    """
    # Cargar audio en mono a sample_rate
    y, sr = librosa.load(file_path, sr=sample_rate, mono=True)

    # Si el audio está vacío por algún problema, devolver dict vacío
    if len(y) == 0:
        print(f"[WARN] Audio vacío: {file_path}")
        return {}

    feats = {}

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    for i in range(n_mfcc):
        coeff = mfcc[i]
        feats[f"mfcc_{i+1}_mean"] = float(np.mean(coeff))
        feats[f"mfcc_{i+1}_std"]  = float(np.std(coeff))

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    feats["zcr_mean"] = float(np.mean(zcr))
    feats["zcr_std"]  = float(np.std(zcr))

    # RMS (energía)
    rms = librosa.feature.rms(y=y)[0]
    feats["rms_mean"] = float(np.mean(rms))
    feats["rms_std"]  = float(np.std(rms))

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    feats["centroid_mean"] = float(np.mean(centroid))
    feats["centroid_std"]  = float(np.std(centroid))

    # Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    feats["bandwidth_mean"] = float(np.mean(bandwidth))
    feats["bandwidth_std"]  = float(np.std(bandwidth))

    return feats


def build_dataset(
    audio_dir: str,
    output_csv: str,
    sample_rate: int = 16000,
    n_mfcc: int = 13
) -> None:
    """
    Recorre audio_dir, extrae features de todos los .wav
    y guarda un CSV con una fila por archivo.

    La etiqueta se extrae del NOMBRE DEL ARCHIVO usando:
    - Desde el inicio hasta antes del primer dígito.
    Ejemplos válidos:
        'proporcion01.wav'  -> 'proporcion'
        'contar_02.wav'     -> 'contar'
        'salir15_prueba.wav'-> 'salir'
    """
    rows = []

    for root, _, files in os.walk(audio_dir):
        for fname in files:
            if not fname.lower().endswith(".wav"):
                continue

            file_path = os.path.join(root, fname)

            # Extraer features
            feats = extract_features_from_file(
                file_path=file_path,
                sample_rate=sample_rate,
                n_mfcc=n_mfcc
            )

            # Si no se pudieron extraer features, omitir el archivo
            if not feats:
                print(f"[WARN] Sin features, se omite: {file_path}")
                continue

            # === NUEVA LÓGICA DE ETIQUETA SIN GUION BAJO ===
            # Tomamos todas las letras desde el inicio hasta antes del primer número.
            base = os.path.basename(file_path)
            m = re.match(r"[A-Za-zñÑáéíóúÁÉÍÓÚ]+", base)

            if m:
                label = m.group(0).lower()
            else:
                print(f"[WARN] No se pudo extraer etiqueta de '{base}', se omite.")
                continue
            # ================================================

            row = {
                "file_path": file_path,
                "label": label
            }
            row.update(feats)
            rows.append(row)
            print(f"[OK] Features extraídas de {file_path}")

    if not rows:
        print("[WARN] No se generaron filas. Revisá el directorio de audio.")
        return

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[OK] CSV guardado en: {output_csv}")


if __name__ == "__main__":
    # Usamos los audios YA PREPROCESADOS
    AUDIO_DIR = "data_voz/interm"
    OUTPUT_CSV = "data_voz/features/features_audio.csv"

    build_dataset(AUDIO_DIR, OUTPUT_CSV)
