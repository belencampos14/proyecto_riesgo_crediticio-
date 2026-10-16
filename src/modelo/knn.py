"""Clasificador KNN vectorizado usando NumPy."""

import numpy as np

from src.modelo.distancias import calcular_matriz_distancias


def obtener_vecinos_cercanos(
    distancias: np.ndarray,
    k: int,
) -> np.ndarray:
    """Obtiene los índices de los k vecinos más cercanos."""
    return np.argsort(distancias, axis=1)[:, :k]


def votar_clase_mayoritaria(
    etiquetas_vecinos: np.ndarray,
) -> int:
    """Devuelve la clase más frecuente; en empate, la menor."""
    return int(
        np.argmax(np.bincount(etiquetas_vecinos))
    )


def predecir_knn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """Predice las etiquetas mediante votación mayoritaria.

    Args:
        x_train: Matriz de entrenamiento con forma (M, D).
        y_train: Etiquetas enteras no negativas con forma (M,).
        x_test: Matriz de prueba con forma (N, D).
        k: Cantidad de vecinos.

    Returns:
        Vector de predicciones con forma (N,).
    """
    if not isinstance(k, (int, np.integer)):
        raise ValueError("k debe ser un número entero.")

    if not 1 <= k <= x_train.shape[0]:
        raise ValueError(
            "k debe estar entre 1 y las filas de entrenamiento."
        )

    distancias = calcular_matriz_distancias(
        x_train,
        x_test,
    )
    posiciones = obtener_vecinos_cercanos(distancias, k)
    etiquetas_vecinos = y_train[posiciones]

    return np.apply_along_axis(
        votar_clase_mayoritaria,
        axis=1,
        arr=etiquetas_vecinos,
    )