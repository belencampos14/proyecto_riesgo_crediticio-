"""Métricas de clasificación calculadas con NumPy."""

import numpy as np


def matriz_confusion(y_real, y_pred):
    """Devuelve TP, FP, TN y FN para etiquetas binarias 0/1."""
    y_real = np.asarray(y_real, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    conteos = np.bincount(
        2 * y_real + y_pred,
        minlength=4,
    )

    tn, fp, fn, tp = (
        int(valor) for valor in conteos
    )

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }


def precision(y_real, y_pred):
    """Calcula Precision para la clase positiva 1."""
    conteos = matriz_confusion(y_real, y_pred)
    denominador = conteos["TP"] + conteos["FP"]

    return (
        conteos["TP"] / denominador
        if denominador
        else 0.0
    )


def recall(y_real, y_pred):
    """Calcula Recall para la clase positiva 1."""
    conteos = matriz_confusion(y_real, y_pred)
    denominador = conteos["TP"] + conteos["FN"]

    return (
        conteos["TP"] / denominador
        if denominador
        else 0.0
    )


def f1_score(y_real, y_pred):
    """Calcula F1 para la clase positiva 1."""
    valor_precision = precision(y_real, y_pred)
    valor_recall = recall(y_real, y_pred)
    denominador = valor_precision + valor_recall

    return (
        2 * valor_precision * valor_recall / denominador
        if denominador
        else 0.0
    )


def resumen_metricas(y_real, y_pred):
    """Devuelve conteos y métricas globales."""
    conteos = matriz_confusion(y_real, y_pred)
    total = sum(conteos.values())
    aciertos = conteos["TP"] + conteos["TN"]

    return {
        **conteos,
        "precision": precision(y_real, y_pred),
        "recall": recall(y_real, y_pred),
        "f1_score": f1_score(y_real, y_pred),
        "accuracy": aciertos / total if total else 0.0,
    }