"""Codificación de categorías sin modificar los datos originales."""

import pandas as pd


def codificar_binaria(
    datos: pd.DataFrame,
    columna: str,
    valor_positivo: str,
) -> pd.DataFrame:
    """Representa la categoría positiva con uno y las demás con cero."""
    datos_nuevos = datos.copy()
    datos_nuevos[columna] = (
        datos[columna] == valor_positivo
    ).astype(float)

    return datos_nuevos


def codificar_ordinal(
    datos: pd.DataFrame,
    columna: str,
    orden: list,
) -> pd.DataFrame:
    """Convierte categorías ordenadas en valores entre cero y uno."""
    posiciones = dict(zip(orden, range(len(orden))))
    datos_nuevos = datos.copy()
    datos_nuevos[columna] = (
        datos[columna].map(posiciones) / (len(orden) - 1)
    )

    return datos_nuevos


def codificar_one_hot(
    datos: pd.DataFrame,
    columna: str,
) -> pd.DataFrame:
    """Reemplaza una columna categórica por columnas binarias."""
    columnas_nuevas = pd.get_dummies(
        datos[columna],
        prefix=columna,
        dtype=float,
    )

    return pd.concat(
        [datos.drop(columns=[columna]), columnas_nuevas],
        axis=1,
    )


def dataframe_a_tuplas(datos: pd.DataFrame) -> tuple:
    """Convierte una tabla numérica en una tupla de filas."""
    return tuple(
        map(
            tuple,
            datos.itertuples(index=False, name=None),
        )
    )