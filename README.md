# Sign-Language-MNIST

**Asignatura:** TLY1102 – Técnicas Avanzadas de Machine Learning I\
**Equipo de Trabajo:** Rodrigo Aedo - Benjamín Figueroa - Martina López\
**Repositorio:** https://github.com/rodrigo0187/Sign-Language-MNIST.git

---
## 1. Descripción del Problema de Negocio
Las personas sordas o con dificultades auditivas se comunican en muchos casos mediante lenguaje de señas. Contar con sistemas automáticos capaces de reconocer señas a partir de imágenes es un primer paso hacia herramientas de traducción en tiempo real que faciliten la comunicación entre personas sordas y oyentes que no conocen la lengua de señas.

## 2. Objetivos del Proyecto
- Implementar un modelo de **Perceptrón Multicapa (MLP)** capaz de clasificar imágenes.
- Aplicar un flujo completo de trabajo en aprendizaje supervisado: carga de datos, preprocesamiento, definición de arquitectura, entrenamiento, validación y evaluación.
- Evaluar el desempeño del modelo con métricas estándar de clasificación e interpretar los resultados de forma crítica.
- Identificar las limitaciones del uso de un MLP para tareas de clasificación de imágenes.

## 3. Definición de KPIs que resolverán el problema de negocio.
| KPI | Descripción | Umbral Aceptable | Justificación en el negocio |
| :--- | :---- | :--- | :--- | 
| **Accuracy** | Porcentaje total de imágenes correctamente clasificadas. | **$\ge 85\%$** | Con este porcentaje se asegura que la mayoría de caracteres traducidos sean correctos. |
| **Precision** | Porcentaje de predicciones correctas de una letra que realmente corresponden a esa letra | **$\ge 80\$** | Con esto se minimiza los falsos positivos al interpretar un gesto. |
| **Recall** | Capacidad del modelo para identificar instancias reales de una seña | **$\ge 00\%** | Garantiza que el sistema no pase por alto letras claves. |
| **F1-Score** | Mide el equilibrio entre Precision y Recall | **$\ge 0.82$** | Balancea la precisión y exhaustividad en clases con ligera variación de muestras. |
| **Matriz de confusión** | Porcentaje de confusión entre letras similares | 15%? | Permite identificar y mitigar errores entre letras parecidas. |

## 4. Descripción de las fuentes de datos utilizadas
El proyecto utiliza el conjunto de datos de **Sign Language MNIST**, una adaptacion del abcedario de señas americano en formato de imagenes en escala de grises.
* **Origen y formato:**
    * **Archivos csv:** De entrenamiento 'sign_mnist_train.csv' y de prueba 'sign_mnist_test.csv'.
    * **Estructura:** Cada fila es una imagen aplanada de $28 \times 28$ pixeles, que equivale a $784$ pixeles ('pixel1' a 'pixel784'), mas la columna objetivo **label**.
* **Volumen de datos:**
    * **Entrenamiento:** $27.455$ muestras.
    * **Prueba:** $7.172$ muestras.
    * **Total:** $34.627$ muestras.
* **Distribución de clases:**
    * Contiene 24 clases numéricas (valores de $0$ a $24$).
    * **Observación técnica:** Se excluye la clase $9$ que vendría siendo la letra **J** y la clase $25$ que es la **Z**, esto es porque ambas señas requieren un gesto con movimiento, por lo que no puede representarse en una imagen estática. 

## 5. Metodología CRISP-DM

El proyecto esta bajo el marco de trabajo de **CRISP-DM**, el cual se encuentra estructurado en sus 6 fases principales:

`[1. Comprensión del Negocio]` ➔ `[2. Comprensión de los Datos]` ➔ `[3. Preparación de los Datos]` ➔ `[4. Modelado]` ➔ `[5. Evaluación]` ➔ `[6. Despliegue]`

* **Comprensión del negocio:**
Identificación del problema de entendimiento de lengua de señas. Definición de objetivos técnicos y KPIs de precisión y error.
* **Comprensión de los datos:**
	Exploración inicial del dataset, verificación del tamaño de muestra ($34.627$), revisión de equilibrio de clases y formato de pixeles.
* **Preparación de los datos:**
	Normalización de intensidad $[0,1]$, conversión a ‘tf.float32’ y división del conjunto en **Entrenamiento**, **Validación** y **Prueba**.
* **Modelado:**
	Construcción de la red neuronal multicapa (MLP) en Tensor Flow, definiendo capas densas, funciones de activación, regularización y el optimizador.
* **Evaluación:**
Análisis del rendimiento del modelo mediante los KPIs definidos.
* **Despliegue:**
Documentación técnica “**README.md**”, versionamiento de código y entrega de la solución reproducible.


## 6. Análisis Exploratorio EDA

> En primera instancia se procede a aplicar una metodologia para analisis exploratorio de datos aplicando **EDA**.

### Librerías utilizadas

> Se utilizaron las siguientes librerias como refuerzo para este analisis:

- **pandas:** exploracion de estructuras, nulos, tipos, estadistica.
- **numpy:** analizar los valores y operaciones numericas.
- **matplotlib, seaborn :** para una visualizacion exploratoria como distribuciones y relaciones.
- De las cuales se utilizaron `shape`, `describe`, `info`, `unique`, etc.

> Primera fase se analiza de la siguiente manera:

- Conocimiento del dataset.
- Conocer las dimensiones de ambos archivos `train` y `test`.
- Conocer las clases y su cantidad.
- Revisar valores.
- Entender `features` y `label`.

### Procesamiento
- Separación de datos en características `features` y variable objetivo `label`.
- Normalización de pixeles mediante escalamiento al rango $[0,1]$.
- Conversión de tipos de datos a tensores ('tf.float32') para optimizar.

### Diagnóstico
- La intensidad de pixeles varia en el rango de $[0, 255]$. Para evitar una saturación de las funciones de activación, se define la normalización al intervalo $[0,1]$.
- Letras como la **M**, **N** presentan formas casi idénticas al reducirse a $28 \times 28$ píxeles, que solo se diferencian por pequeños cambios de posición del pulgar.
- Existen variaciones de brillo en los datos, lo que obliga al MLP a aprender bordes.

### Preparación de 1 pipeline: tensores a Dataset

- En la etapa de normalización se utiliza la función normalizar. En esta fase ocurren dos procesos: primero se realiza el casteo de los valores numéricos, convirtiendo los datos a float32, lo que permite trabajar con ellos como tensores de TensorFlow. Posteriormente, cada valor de intensidad de los píxeles se divide por 255, que corresponde al valor máximo de intensidad, obteniendo valores dentro del rango de 0 a 1.

- Posteriormente, los tensores de características y sus etiquetas se utilizan para construir un tf.data.Dataset, manteniendo la correspondencia entre cada imagen y su etiqueta.
