# preprocess_audio.py

import os
import numpy as np
import librosa
import soundfile as sf

def preprocess_audio_file(
    input_path: str,
    output_path: str,
    sample_rate: int = 16000,
    top_db: int = 20,
    preemphasis_coef: float = 0.97
) -> None:
    """
    Carga un audio, recorta silencios, normaliza amplitud,
    aplica preénfasis y guarda el resultado.
    """
    # Cargar en mono y resamplear si hace falta
    y, sr = librosa.load(input_path, sr=sample_rate, mono=True)

    # Recortar silencios
    y_trim, _ = librosa.effects.trim(y, top_db=top_db)

    if y_trim.size == 0:
        print(f"[WARN] Audio vacío después de trim: {input_path}")
        return

    # Normalizar amplitud
    max_val = np.max(np.abs(y_trim)) + 1e-9
    y_norm = y_trim / max_val

    # Preénfasis
    y_pre = librosa.effects.preemphasis(y_norm, coef=preemphasis_coef)

    # Crear carpeta destino si no existe
    #os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dirpath = os.path.dirname(output_path)
    if dirpath:  # Solo crear si no es cadena vacía
        os.makedirs(dirpath, exist_ok=True)


    # Guardar audio preprocesado
    sf.write(output_path, y_pre, sample_rate)
    print(f"[OK] Preprocesado: {input_path} -> {output_path}")


def preprocess_directory(
    input_dir: str,
    output_dir: str,
    sample_rate: int = 16000
) -> None:
    """
    Recorre input_dir y preprocesa todos los .wav,
    guardando la misma estructura en output_dir.
    """
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if not fname.lower().endswith(".wav"):
                continue

            in_path = os.path.join(root, fname)

            # Preservar estructura relativa
            rel_path = os.path.relpath(in_path, input_dir)
            out_path = os.path.join(output_dir, rel_path)

            preprocess_audio_file(in_path, out_path, sample_rate=sample_rate)


if __name__ == "__main__":
    # Ajusta estas rutas a tu proyecto
    INPUT_DIR = "data_voz/raso"
    OUTPUT_DIR = "data_voz/interm"

    preprocess_directory(INPUT_DIR, OUTPUT_DIR)
