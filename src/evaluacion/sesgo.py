"""Herramientas para auditoría exploratoria por grupos."""

import numpy as np
import pandas as pd

from src.evaluacion.metricas import resumen_metricas


def metricas_por_grupo(y_real, y_pred, grupos):
    """Calcula las métricas de clasificación para cada grupo."""
    y_real = np.asarray(y_real, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    grupos = np.asarray(grupos)

    filas = []

    for grupo in pd.unique(grupos):
        mascara = grupos == grupo

        fila = {
            "grupo": grupo,
            "n": int(mascara.sum()),
        }

        fila.update(
            resumen_metricas(
                y_real[mascara],
                y_pred[mascara],
            )
        )

        filas.append(fila)

    return pd.DataFrame(filas)


def disparate_impact(
    y_pred,
    grupos,
    etiqueta_protegido="menor_25",
    etiqueta_control="25_o_mas",
):
    """Divide la tasa favorable protegida entre la tasa de control.

    La predicción cero se utiliza como proxy exploratorio de
    resultado favorable, no como aprobación bancaria real.
    """
    y_pred = np.asarray(y_pred, dtype=int)
    grupos = np.asarray(grupos)

    protegido = grupos == etiqueta_protegido
    control = grupos == etiqueta_control

    tasa_protegido = (
        float(np.mean(y_pred[protegido] == 0))
        if protegido.any()
        else np.nan
    )

    tasa_control = (
        float(np.mean(y_pred[control] == 0))
        if control.any()
        else np.nan
    )

    ratio = (
        tasa_protegido / tasa_control
        if np.isfinite(tasa_protegido)
        and np.isfinite(tasa_control)
        and tasa_control > 0
        else np.nan
    )

    return {
        "grupo_protegido": etiqueta_protegido,
        "grupo_control": etiqueta_control,
        "tasa_favorable_protegido": tasa_protegido,
        "tasa_favorable_control": tasa_control,
        "disparate_impact": ratio,
    }