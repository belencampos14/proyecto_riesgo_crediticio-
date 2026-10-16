"""Conversión de colecciones inmutables a arreglos NumPy."""

import numpy as np


def tuplas_a_matriz(filas: tuple) -> np.ndarray:
    """Convierte filas numéricas en una matriz de solo lectura."""
    matriz = np.array(filas, dtype=float)

    if matriz.ndim != 2:
        raise ValueError(
            "Los datos deben formar una matriz de dos dimensiones."
        )

    matriz.setflags(write=False)
    return matriz


def tuplas_a_vector(valores: tuple) -> np.ndarray:
    """Convierte etiquetas enteras en un vector de solo lectura."""
    vector = np.array(valores, dtype=int)

    if vector.ndim != 1:
        raise ValueError(
            "Las etiquetas deben formar un vector de una dimensión."
        )

    vector.setflags(write=False)
    return vector