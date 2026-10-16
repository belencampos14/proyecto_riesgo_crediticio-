"""Guardado del modelo KNN y su configuración."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np


def guardar_modelo(
    x_train,
    y_train,
    configuracion,
    carpeta_modelos,
):
    """Guarda el entrenamiento y la configuración del modelo.

    Escribe archivos y comprueba que los arreglos guardados
    se recuperan sin cambios.
    """
    identificador = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    carpeta = Path(carpeta_modelos) / f"knn_{identificador}"
    carpeta.mkdir(parents=True, exist_ok=False)

    ruta_modelo = carpeta / "modelo.npz"
    ruta_configuracion = carpeta / "configuracion.json"

    np.savez_compressed(
        ruta_modelo,
        x_train=x_train,
        y_train=y_train,
    )

    ruta_configuracion.write_text(
        json.dumps(
            configuracion,
            ensure_ascii=False,
            indent=4,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    with np.load(ruta_modelo, allow_pickle=False) as modelo:
        if not np.array_equal(modelo["x_train"], x_train):
            raise ValueError("La matriz guardada no coincide.")

        if not np.array_equal(modelo["y_train"], y_train):
            raise ValueError("Las etiquetas guardadas no coinciden.")

    return carpeta