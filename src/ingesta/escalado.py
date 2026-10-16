"""Escalado Min-Max funcional sin modificar las entradas."""

from functools import reduce

import pandas as pd


def calcular_parametros_min_max(
    datos: pd.DataFrame,
    columnas: list,
) -> tuple:
    """Obtiene mínimos y máximos de las columnas de entrenamiento."""
    faltantes = list(
        filter(
            lambda columna: columna not in datos.columns,
            columnas,
        )
    )

    if faltantes:
        raise ValueError(f"Columnas inexistentes: {faltantes}")

    return tuple(
        map(
            lambda columna: (
                columna,
                datos[columna].min(),
                datos[columna].max(),
            ),
            columnas,
        )
    )


def escalar_valores_min_max(
    valores: pd.Series,
    minimo=None,
    maximo=None,
) -> list:
    """Escala valores usando parámetros recibidos o locales."""
    if minimo is None:
        minimo = valores.min()

    if maximo is None:
        maximo = valores.max()

    if pd.isna(minimo) or pd.isna(maximo):
        raise ValueError(
            "No se puede escalar una columna sin datos válidos."
        )

    if minimo == maximo:
        return list(map(lambda valor: 0.0, valores))

    return list(
        map(
            lambda valor: (valor - minimo) / (maximo - minimo),
            valores,
        )
    )


def escalar_columna_min_max(
    datos: pd.DataFrame,
    columna: str,
    minimo=None,
    maximo=None,
) -> pd.DataFrame:
    """Devuelve una nueva tabla con una columna escalada."""
    valores = escalar_valores_min_max(
        datos[columna],
        minimo,
        maximo,
    )

    return datos.assign(**{columna: valores})


def aplicar_parametros_min_max(
    datos: pd.DataFrame,
    parametros: tuple,
) -> pd.DataFrame:
    """Aplica parámetros sin volver a calcular mínimos y máximos."""
    return reduce(
        lambda resultado, parametro: escalar_columna_min_max(
            resultado,
            parametro[0],
            parametro[1],
            parametro[2],
        ),
        parametros,
        datos.copy(),
    )


def escalar_columnas_min_max(
    datos: pd.DataFrame,
    columnas: list,
) -> pd.DataFrame:
    """Calcula y aplica Min-Max al conjunto recibido."""
    parametros = calcular_parametros_min_max(datos, columnas)

    return aplicar_parametros_min_max(datos, parametros)