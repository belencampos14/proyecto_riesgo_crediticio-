"""Cálculo vectorizado de distancias euclídeas."""

import numpy as np


def calcular_matriz_distancias(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> np.ndarray:
    """Calcula distancias entre prueba y entrenamiento sin bucles.

    Args:
        x_train: Matriz de entrenamiento con forma (M, D).
        x_test: Matriz de prueba con forma (N, D).

    Returns:
        Matriz de distancias con forma (N, M).
    """
    suma_test = np.sum(
        x_test ** 2,
        axis=1,
        keepdims=True,
    )
    suma_train = np.sum(x_train ** 2, axis=1)
    producto = x_test @ x_train.T

    distancias_cuadradas = suma_test - 2 * producto + suma_train

    return np.sqrt(
        np.maximum(distancias_cuadradas, 0.0)
    )
