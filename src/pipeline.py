"""Pipeline de riesgo crediticio con KNN y auditoría de resultados."""

from pathlib import Path

import numpy as np

from src.evaluacion.metricas import resumen_metricas
from src.evaluacion.sesgo import disparate_impact, metricas_por_grupo
from src.ingesta.codificacion import (
    codificar_binaria,
    codificar_one_hot,
    codificar_ordinal,
    dataframe_a_tuplas,
)
from src.ingesta.escalado import (
    aplicar_parametros_min_max,
    calcular_parametros_min_max,
)
from src.ingesta.lectura import cargar_dataset
from src.ingesta.limpieza import imputar_con_parametros, pipe
from src.ingesta.outliers import filtrar_outliers_iqr
from src.modelo.knn import predecir_knn
from src.modelo.persistencia import guardar_modelo
from src.modelo.transformacion import tuplas_a_matriz, tuplas_a_vector


RAIZ_PROYECTO = Path(__file__).resolve().parent.parent

RUTA_DATOS = (
    RAIZ_PROYECTO
    / "data"
    / "raw"
    / "credit_risk_dataset.csv"
)

CARPETA_MODELOS = RAIZ_PROYECTO / "models"

SEMILLA = 42
K_VECINOS = 5
TAMANO_LOTE = 256


def codificar_datos(datos):
    """Codifica categorías sin modificar la tabla recibida."""
    return pipe(
        datos,
        lambda tabla: codificar_binaria(
            tabla,
            "cb_person_default_on_file",
            "Y",
        ),
        lambda tabla: codificar_ordinal(
            tabla,
            "loan_grade",
            list("ABCDEFG"),
        ),
        lambda tabla: codificar_one_hot(
            tabla,
            "person_home_ownership",
        ),
        lambda tabla: codificar_one_hot(
            tabla,
            "loan_intent",
        ),
    )


def preparar_datos(
    datos_train,
    datos_test,
    devolver_parametros=False,
):
    """Prepara ambos conjuntos con estadísticas de entrenamiento."""
    columnas_nulos = (
        "person_emp_length",
        "loan_int_rate",
    )
    columnas_escalado = [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
    ]

    entrenamiento = datos_train.copy()
    prueba = datos_test.copy()

    # Calculamos las medianas únicamente con entrenamiento.
    parametros_imputacion = tuple(
        map(
            lambda columna: (
                columna,
                entrenamiento[columna].median(),
            ),
            columnas_nulos,
        )
    )

    # Reutilizamos las mismas medianas en ambos conjuntos.
    entrenamiento = imputar_con_parametros(
        entrenamiento,
        parametros_imputacion,
    )
    prueba = imputar_con_parametros(
        prueba,
        parametros_imputacion,
    )

    # Filtramos outliers únicamente en entrenamiento.
    # Conservamos la prueba completa para evaluar casos extremos.
    entrenamiento = filtrar_outliers_iqr(
        entrenamiento,
        "person_emp_length",
    )

    if entrenamiento.empty:
        raise ValueError("El entrenamiento quedó sin registros.")

    # Calculamos Min-Max únicamente con entrenamiento filtrado.
    parametros_escalado = calcular_parametros_min_max(
        entrenamiento,
        columnas_escalado,
    )

    entrenamiento = aplicar_parametros_min_max(
        entrenamiento,
        parametros_escalado,
    )
    prueba = aplicar_parametros_min_max(
        prueba,
        parametros_escalado,
    )

    entrenamiento = codificar_datos(entrenamiento)
    prueba = codificar_datos(prueba)

    # Conservamos las mismas columnas y su orden.
    # Categorías exclusivas de prueba quedan sin indicador propio.
    prueba = prueba.reindex(
        columns=entrenamiento.columns,
        fill_value=0.0,
    )

    if devolver_parametros:
        parametros = {
            "imputacion": [
                {
                    "columna": columna,
                    "mediana": float(mediana),
                }
                for columna, mediana in parametros_imputacion
            ],
            "escalado_min_max": [
                {
                    "columna": columna,
                    "minimo": float(minimo),
                    "maximo": float(maximo),
                }
                for columna, minimo, maximo in parametros_escalado
            ],
        }

        return entrenamiento, prueba, parametros

    return entrenamiento, prueba


def predecir_por_lotes(
    x_train,
    y_train,
    x_test,
    k=K_VECINOS,
    tamano_lote=TAMANO_LOTE,
):
    """Predice en lotes para controlar el consumo de memoria."""
    if tamano_lote < 1:
        raise ValueError("El tamaño del lote debe ser positivo.")

    predicciones = []

    for inicio in range(0, len(x_test), tamano_lote):
        fin = inicio + tamano_lote

        predicciones.append(
            predecir_knn(
                x_train,
                y_train,
                x_test[inicio:fin],
                k=k,
            )
        )

    if not predicciones:
        return np.array([], dtype=int)

    return np.concatenate(predicciones)


def ejecutar_pipeline(
    ruta_datos=RUTA_DATOS,
    mostrar=True,
    exportar_modelo=True,
):
    """Ejecuta preparación, predicción, auditoría y exportación.

    Args:
        ruta_datos: Ubicación del CSV original.
        mostrar: Indica si se imprimen los resultados.
        exportar_modelo: Indica si se guarda el modelo en disco.

    Returns:
        Diccionario con resultados y, si se exporta, ruta del modelo.
    """
    datos_originales = cargar_dataset(str(ruta_datos))

    if len(datos_originales) < 2:
        raise ValueError("Se necesitan al menos dos registros.")

    # Partición reproducible antes del preprocesamiento.
    np.random.seed(SEMILLA)
    posiciones = np.random.permutation(len(datos_originales))
    corte = int(0.8 * len(datos_originales))

    idx_train = posiciones[:corte]
    idx_test = posiciones[corte:]

    datos_train = datos_originales.iloc[idx_train].copy()
    datos_test = datos_originales.iloc[idx_test].copy()

    # Conservamos las edades originales en el orden de prueba.
    edad_test = datos_test["person_age"].to_numpy(copy=True)

    entrenamiento, prueba, parametros = preparar_datos(
        datos_train,
        datos_test,
        devolver_parametros=True,
    )

    # La edad se utiliza para auditoría y se excluye del clasificador.
    columnas_excluidas = ["loan_status", "person_age"]

    caracteristicas_train = entrenamiento.drop(
        columns=columnas_excluidas
    )
    caracteristicas_test = prueba.drop(
        columns=columnas_excluidas
    )

    x_train = tuplas_a_matriz(
        dataframe_a_tuplas(caracteristicas_train)
    )
    x_test = tuplas_a_matriz(
        dataframe_a_tuplas(caracteristicas_test)
    )

    y_train = tuplas_a_vector(
        tuple(entrenamiento["loan_status"])
    )
    y_test = tuplas_a_vector(
        tuple(prueba["loan_status"])
    )

    if not np.isfinite(x_train).all():
        raise ValueError("El entrenamiento contiene nulos o infinitos.")

    if not np.isfinite(x_test).all():
        raise ValueError("La prueba contiene nulos o infinitos.")

    y_pred = predecir_por_lotes(
        x_train,
        y_train,
        x_test,
    )

    metricas = resumen_metricas(y_test, y_pred)

    grupos_edad = np.where(
        edad_test < 25,
        "menor_25",
        "25_o_mas",
    )

    tabla_grupos = metricas_por_grupo(
        y_test,
        y_pred,
        grupos_edad,
    )
    resultado_di = disparate_impact(
        y_pred,
        grupos_edad,
    )

    resultados_completos = {
        "filas_originales": int(len(datos_originales)),
        "filas_entrenamiento_inicial": int(len(datos_train)),
        "filas_outliers_retiradas": int(
            len(datos_train) - len(entrenamiento)
        ),
        "filas_evaluadas": int(len(y_train) + len(y_test)),
        "filas_entrenamiento": int(len(y_train)),
        "filas_prueba": int(len(y_test)),
        "semilla": SEMILLA,
        "k": K_VECINOS,
        "metricas_globales": metricas,
        "metricas_por_grupo": tabla_grupos,
        "disparate_impact": resultado_di,
    }

    if exportar_modelo:
        configuracion = {
            "algoritmo": "KNN",
            "semilla": SEMILLA,
            "k": K_VECINOS,
            "tamano_lote": TAMANO_LOTE,
            "columnas_caracteristicas": list(
                caracteristicas_train.columns
            ),
            "columnas_excluidas": columnas_excluidas,
            "preprocesamiento": parametros,
            "codificacion": {
                "binaria": {
                    "columna": "cb_person_default_on_file",
                    "valor_positivo": "Y",
                },
                "ordinal": {
                    "columna": "loan_grade",
                    "orden": list("ABCDEFG"),
                },
                "one_hot": [
                    "person_home_ownership",
                    "loan_intent",
                ],
            },
            "outliers_entrenamiento": {
                "metodo": "IQR",
                "columna": "person_emp_length",
                "factor": 1.5,
                "aplicar_a_nuevos_datos": False,
            },
        }

        carpeta_modelo = guardar_modelo(
            x_train,
            y_train,
            configuracion,
            CARPETA_MODELOS,
        )

        resultados_completos["carpeta_modelo"] = str(
            carpeta_modelo
        )

    if mostrar:
        print(f"Filas originales: {len(datos_originales)}")
        print("Partición inicial: 80 % entrenamiento / 20 % prueba")
        print(f"Entrenamiento inicial: {len(datos_train)}")
        print(f"Entrenamiento después de IQR: {len(y_train)}")
        print(f"Registros de prueba evaluados: {len(y_test)}")
        print(f"Semilla: {SEMILLA} | k: {K_VECINOS}")

        print("\nMétricas globales:")
        for clave, valor in metricas.items():
            print(f"{clave}: {valor}")

        print("\nMétricas por grupo de edad:")
        print(tabla_grupos.to_string(index=False))

        print("\nDisparate Impact:")
        for clave, valor in resultado_di.items():
            print(f"{clave}: {valor}")

        if exportar_modelo:
            print(f"\nModelo guardado en: {carpeta_modelo}")

        print(
            "\nNota: la predicción 0 es un proxy exploratorio "
            "de resultado favorable, no una aprobación bancaria real."
        )

    return resultados_completos


if __name__ == "__main__":
    ejecutar_pipeline()