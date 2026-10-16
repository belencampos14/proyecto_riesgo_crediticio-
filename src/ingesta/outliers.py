"""Funciones puras para detectar y filtrar outliers mediante IQR."""

import pandas as pd


def calcular_limites_iqr(
    datos: pd.DataFrame,
    columna: str,
) -> tuple:
    """Calcula los límites inferior y superior mediante IQR."""
    q1 = datos[columna].quantile(0.25)
    q3 = datos[columna].quantile(0.75)
    rango_iqr = q3 - q1

    limite_inferior = q1 - 1.5 * rango_iqr
    limite_superior = q3 + 1.5 * rango_iqr

    return limite_inferior, limite_superior


def detectar_outliers_iqr(
    datos: pd.DataFrame,
    columna: str,
) -> pd.DataFrame:
    """Devuelve una copia de las filas fuera de los límites."""
    limite_inferior, limite_superior = calcular_limites_iqr(
        datos,
        columna,
    )

    filtro = (
        (datos[columna] < limite_inferior)
        | (datos[columna] > limite_superior)
    )

    return datos.loc[filtro].copy()


def filtrar_outliers_iqr(
    datos: pd.DataFrame,
    columna: str,
) -> pd.DataFrame:
    """Devuelve una copia de las filas dentro de los límites."""
    limite_inferior, limite_superior = calcular_limites_iqr(
        datos,
        columna,
    )

    filtro = (
        (datos[columna] >= limite_inferior)
        & (datos[columna] <= limite_superior)
    )

    return datos.loc[filtro].copy()