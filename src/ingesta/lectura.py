"""Funciones para leer el dataset."""

import pandas as pd


def cargar_dataset(ruta_archivo: str) -> pd.DataFrame:
    """Lee un CSV sin modificar el archivo original."""
    return pd.read_csv(ruta_archivo)