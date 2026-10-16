"""Funciones puras de imputación y composición."""

from functools import reduce

import pandas as pd


def completar_con_valor(
    datos: pd.DataFrame,
    columna: str,
    valor: float,
) -> pd.DataFrame:
    """Completa nulos sin modificar la tabla recibida."""
    if pd.isna(valor):
        raise ValueError("El valor de imputación no puede ser nulo.")

    return datos.assign(
        **{columna: datos[columna].fillna(valor)}
    )


def imputar_mediana(
    datos: pd.DataFrame,
    columna: str,
) -> pd.DataFrame:
    """Completa nulos con la mediana del conjunto recibido."""
    mediana = datos[columna].median()

    return completar_con_valor(
        datos,
        columna,
        mediana,
    )


def imputar_con_parametros(
    datos: pd.DataFrame,
    parametros: tuple,
) -> pd.DataFrame:
    """Encadena imputaciones con valores calculados previamente."""
    return reduce(
        lambda resultado, parametro: completar_con_valor(
            resultado,
            parametro[0],
            parametro[1],
        ),
        parametros,
        datos.copy(),
    )


def pipe(datos: pd.DataFrame, *funciones) -> pd.DataFrame:
    """Aplica las funciones en orden y devuelve el resultado."""
    return reduce(
        lambda resultado, funcion: funcion(resultado),
        funciones,
        datos,
    )