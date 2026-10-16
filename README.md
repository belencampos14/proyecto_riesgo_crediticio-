# Proyecto de riesgo crediticio

Proyecto académico que integra programación funcional, cálculo
vectorizado con NumPy y auditoría de resultados mediante un
clasificador K-Nearest Neighbors implementado desde cero.

## Estructura

- `data/raw/`: dataset original `credit_risk_dataset.csv`.
- `data/processed/`: carpeta destinada a datos procesados.
- `models/`: modelos exportados en subcarpetas identificadas por fecha.
- `src/ingesta/`: lectura, imputación, filtrado IQR, escalado y codificación.
- `src/modelo/`: conversión a NumPy, distancias, KNN y persistencia.
- `src/evaluacion/`: métricas de clasificación y análisis por grupos.
- `notebooks/auditoria.ipynb`: demostraciones y auditoría de resultados.
- `main.py`: coordina la preparación, predicción, auditoría y exportación.
- `requirements.txt`: dependencias con versiones exactas.
- `.gitignore`: exclusiones para el control de versiones.
- `README.md`: documentación e instrucciones de ejecución.

Actualmente, la coordinación del pipeline permanece en `main.py`.
Los módulos de transformación, clasificación, evaluación y persistencia
se encuentran en `src/`.

## Dataset

Antes de ejecutar el proyecto, colocar el archivo original en:

`data/raw/credit_risk_dataset.csv`

La carpeta `data/` está excluida de Git. Al obtener el proyecto desde
un repositorio, el CSV debe conseguirse por separado y colocarse
en esa ubicación.

El pipeline no modifica el archivo original ni exporta automáticamente
tablas a `data/processed/`.

## Entorno de ejecución

El entorno se creó con Python 3.13.9 en Windows.
Las dependencias se registraron desde el entorno virtual del proyecto.

Para recrearlo, utilizar Python 3.13.9 y abrir una terminal en la raíz
del proyecto: la carpeta que contiene `main.py` y `requirements.txt`.

Comprobar la versión:

```powershell
python --version
```

Debe mostrar Python 3.13.9. Después, crear el entorno e instalar
las dependencias:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

La instalación requiere acceso a las dependencias indicadas
en `requirements.txt`.

La carpeta `.venv` se genera localmente y está excluida de Git.
Cada integrante debe crear su propio entorno; no se incluye
en el ZIP de entrega.

El proyecto actual no utiliza claves de API ni contraseñas,
por lo que no requiere un archivo `.env`.

## Ejecutar el programa principal

Desde la raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe main.py
```

El programa lee el CSV, prepara los datos, realiza las predicciones,
calcula las métricas, compara los grupos de edad y exporta el modelo.

## Abrir el notebook

Registrar el kernel del entorno:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name riesgo_crediticio --display-name "Python (riesgo crediticio)"
```

Iniciar Jupyter desde la raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab
```

Abrir `notebooks/auditoria.ipynb` y seleccionar el kernel
**Python (riesgo crediticio)**.

Ejecutar todas las celdas en orden. La primera celda localiza la raíz
del proyecto para permitir el acceso a los módulos y al dataset.

Después de modificar archivos `.py`, reiniciar el kernel y ejecutar
todas las celdas para cargar los cambios.

## Preprocesamiento funcional

Las transformaciones devuelven nuevas tablas y conservan los datos
recibidos. La lectura del CSV constituye la entrada externa del sistema.

Se utilizan composición de funciones, `map`, `filter` y `reduce`.
Los valores ausentes de `person_emp_length` y `loan_int_rate`
se completan mediante medianas de entrenamiento.

Los outliers de años de empleo se detectan mediante la regla
de 1,5 veces el rango intercuartílico, o IQR.

En la ejecución principal:

1. Se divide el dataset original en entrenamiento y prueba.
2. Se calculan las medianas únicamente con entrenamiento.
3. Se aplican esas medianas a ambos conjuntos.
4. Se filtran los outliers únicamente en entrenamiento.
5. Se calculan los parámetros Min-Max con entrenamiento filtrado.
6. Se aplican esos parámetros a entrenamiento y prueba.
7. Se codifican las variables categóricas y se alinean las columnas.

La prueba se conserva completa para evaluar también casos extremos.

El escalado Min-Max utiliza:

```text
valor_escalado = (valor - minimo) / (maximo - minimo)
```

Las columnas constantes en entrenamiento se transforman en ceros.
Los valores de prueba pueden quedar fuera del intervalo [0, 1]
si superan los límites observados en entrenamiento.

Las variables categóricas se transforman mediante:

- Codificación binaria para `cb_person_default_on_file`.
- Codificación ordinal para `loan_grade`, con orden de A hasta G.
- Codificación one-hot para `person_home_ownership` y `loan_intent`.

Las columnas de prueba se ajustan al orden de entrenamiento.
Una categoría exclusiva de prueba no obtiene un indicador propio.

## Clasificador KNN

El algoritmo de clasificación utiliza NumPy y no emplea scikit-learn.

Las características se convierten de tuplas a matrices bidimensionales.
Las etiquetas se almacenan en vectores. Las conversiones bloquean
la escritura habitual sobre los arreglos resultantes.

Las distancias euclídeas se calculan sin bucles mediante:

```text
distancia(a, b) = sqrt(||a||² - 2(a · b) + ||b||²)
```

Se utilizan broadcasting y multiplicación matricial.
Los pequeños valores negativos producidos por redondeo numérico
se limitan a cero antes de calcular la raíz cuadrada.

La selección y votación utilizan `np.argsort`, `np.bincount`
y `np.apply_along_axis`.

En caso de empate en la votación, gana la etiqueta numérica menor.

La ejecución por lotes limita la memoria utilizada.
El bucle recorre lotes; cada cálculo de distancias es vectorizado.

## Exportación del modelo

Por defecto, cada ejecución completa del pipeline crea una nueva
subcarpeta fechada dentro de `models/`.

Cada exportación contiene:

- `modelo.npz`: matrices de características y etiquetas de entrenamiento.
- `configuracion.json`: configuración del clasificador y parámetros
  necesarios para preparar los datos.

La configuración registra:

- Semilla aleatoria.
- Cantidad de vecinos.
- Tamaño de lote.
- Nombres y orden de las características.
- Columnas excluidas del clasificador.
- Medianas utilizadas para la imputación.
- Mínimos y máximos utilizados para el escalado.
- Reglas de codificación categórica.
- Regla de filtrado IQR aplicada al entrenamiento.

KNN necesita conservar los ejemplos de entrenamiento para realizar
predicciones; no produce coeficientes como una regresión.

Durante la exportación, el programa vuelve a leer `modelo.npz`
y comprueba que sus arreglos coincidan con los originales.

La escritura de archivos se realiza en `src/modelo/persistencia.py`,
separada de las transformaciones puras.

Desde Python o el notebook:

```python
from main import ejecutar_pipeline

resultados = ejecutar_pipeline(mostrar=False)
print(resultados["carpeta_modelo"])
```

`mostrar=False` oculta la impresión de resultados, pero mantiene
la exportación.

Para ejecutar sin guardar otra versión:

```python
resultados = ejecutar_pipeline(
    mostrar=False,
    exportar_modelo=False,
)
```

En ese caso, el diccionario no incluye la clave `carpeta_modelo`.

La exportación conserva el estado del modelo. Actualmente no existe
una función que cargue estos archivos y prepare automáticamente
nuevos registros para inferencia.

## Reproducibilidad

La ejecución principal fija la semilla 42 y realiza una partición
inicial aleatoria de 80 % para entrenamiento y 20 % para prueba.

Después del filtrado IQR, el tamaño de entrenamiento disminuye.
El conjunto de prueba mantiene sus registros originales.

El clasificador utiliza `k=5` y las predicciones se procesan
en lotes de 256 observaciones.

Para comparar ejecuciones deben conservarse:

- El mismo CSV y el mismo orden de sus filas.
- La misma versión del código.
- La misma versión de Python y de las dependencias.
- La misma configuración del pipeline.

Las primeras secciones del notebook son demostraciones de los módulos.
La evaluación final corresponde al pipeline ejecutado desde `main.py`,
que separa entrenamiento y prueba antes del preprocesamiento.

Los nombres de las carpetas exportadas cambian con la fecha y la hora.
Para comprobar reproducibilidad se comparan resultados y configuración,
no los nombres de esas carpetas.

La ejecución desde una copia en otra carpeta está pendiente
de validación.

## Métricas y auditoría

Se calculan TP, FP, TN, FN, Precision, Recall, F1 y Accuracy
para etiquetas binarias 0/1.

Precision, Recall y F1 consideran positiva la clase 1.
Las métricas con denominador cero devuelven 0,0 por convención.

La auditoría compara menores de 25 años con personas de 25 años o más.
Este corte es una elección exploratoria académica, no una clasificación
legal de grupos protegidos.

La edad se conserva en sus unidades originales para la auditoría
y se excluye de las características del clasificador principal.

El Disparate Impact se calcula como:

```text
tasa favorable del grupo menor de 25 / tasa favorable del grupo de control
```

La predicción `loan_status=0` se utiliza como proxy exploratorio de
resultado favorable; no representa una aprobación bancaria real.

Si falta alguno de los grupos o la tasa favorable del grupo de control
es cero, el cálculo devuelve `NaN` porque el cociente no está definido.

La referencia de 0,80 utilizada en el notebook es orientativa.
No demuestra por sí sola presencia o ausencia de discriminación.

## Limitaciones y mitigación propuesta

La exactitud debe analizarse junto con Recall y F1 de la clase 1.
Las diferencias entre grupos también requieren revisar sus tamaños,
errores y tasas favorables.

Se propone revisar la representación de grupos y clases, evaluar
muestreo equilibrado solo en entrenamiento y comparar valores impares
de k mediante validación interna.

La prueba final debe permanecer separada durante esos ajustes.

Estas medidas son propuestas y todavía no se han aplicado.
Los registros con edades extremas requieren revisión adicional.
Excluir la edad no elimina posibles relaciones indirectas con ella.

La interpretación detallada se presenta en
`notebooks/auditoria.ipynb`.

## Control de versiones

El archivo `.gitignore` excluye:

- Los entornos virtuales.
- Los archivos temporales de Python y Jupyter.
- La carpeta `data/`.
- Los formatos binarios de modelos especificados en sus reglas.

El código fuente, el notebook, la documentación y `requirements.txt`
deben conservarse en el control de versiones.

Las reglas actuales excluyen los archivos `.npz` del modelo,
pero no sus archivos `.json` de configuración.

El archivo `.gitignore` no elimina archivos ni retira automáticamente
del seguimiento aquellos que ya estuvieran registrados en Git.

## Compartir el proyecto

Para compartir una copia original con el grupo:

1. Guardar todos los archivos y los resultados del notebook.
2. Crear una copia de la carpeta del proyecto.
3. Excluir de esa copia `.venv`, `__pycache__` e `.ipynb_checkpoints`.
4. Incluir el CSV en `data/raw/`.
5. Incluir los artefactos de `models/` si se desea compartir
   el modelo ya exportado.
6. Comprimir la copia en ZIP.

Cada integrante puede conservar el mismo ZIP como versión original
y trabajar sobre su carpeta extraída.

Para ejecutar el proyecto, cada integrante debe crear su propio entorno
e instalar las dependencias siguiendo este documento.

Las reglas de `.gitignore` solo afectan al seguimiento de Git;
no excluyen archivos al comprimir una carpeta en ZIP.