# Sign-Language-MNIST

**Asignatura:** TLY1102 – Técnicas Avanzadas de Machine Learning I\
**Equipo de Trabajo:** Rodrigo Aedo - Benjamín Figueroa - Martina López\
**Repositorio:** https://github.com/rodrigo0187/Sign-Language-MNIST.git

**site:** https://rodrigo0187.github.io/Sign-Language-MNIST/

---
## 1. Descripción del Problema de Negocio
Las personas sordas o con dificultades auditivas se comunican en muchos casos mediante lenguaje de señas. Contar con sistemas automáticos capaces de reconocer señas a partir de imágenes es un primer paso hacia herramientas de traducción en tiempo real que faciliten la comunicación entre personas sordas y oyentes que no conocen la lengua de señas.

## 2. Objetivos del Proyecto
* **Objetivo general:** Desarrollar y analizar un Modelo de Perceptrón Multicapa (MLP), el cual sea capaz de clasificar imágenes de lengua de señas a partir de pixeles.
* **Objetivos específicos:**
	* Comprender la estructura y las características del conjunto de datos.
	* Aplicar un flujo completo de trabajo en aprendizaje supervisado: carga de datos, preprocesamiento, definición de arquitectura, entrenamiento, validación y evaluación.
	* Evaluar el desempeño del modelo con métricas estándar de clasificación e interpretar los resultados de forma crítica.
	* Identificar las limitaciones del uso de un MLP para tareas de clasificación de imágenes.

## 3. Definición de KPIs que resolverán el problema de negocio.
| KPI | Descripción | Umbral Aceptable | Justificación en el negocio |
| :--- | :---- | :--- | :--- | 
| **Accuracy** | Porcentaje total de imágenes correctamente clasificadas. | **$\ge 85\$%** | Con este porcentaje se asegura que la mayoría de caracteres traducidos sean correctos. |
| **Precision** | Porcentaje de predicciones correctas de una letra que realmente corresponden a esa letra | **$\ge 80\$%** | Con esto se minimiza los falsos positivos al interpretar un gesto. |
| **Recall** | Capacidad del modelo para identificar instancias reales de una seña | **$\ge 80\$%** | Garantiza que el sistema no pase por alto letras claves. |
| **F1-Score** | Mide el equilibrio entre Precision y Recall | **$\ge 0.82$** | Balancea la precisión y exhaustividad en clases con ligera variación de muestras. |
| **Matriz de confusión** | Porcentaje de confusión entre letras similares | **$\le 0.5\$** | Permite identificar y mitigar errores entre letras parecidas. |

## 4. Descripción de las fuentes de datos y trazabilidad
El proyecto utiliza el conjunto de datos de **Sign Language MNIST**, una adaptacion del abcedario de señas americano en formato de imagenes en escala de grises.
* **Origen y formato:**
    * **Archivos csv:** De entrenamiento 'sign_mnist_train.csv' y de prueba 'sign_mnist_test.csv'.
    * **Estructura:** Cada fila es una imagen aplanada de $28 \times 28$ pixeles, que equivale a $784$ pixeles ('pixel1' a 'pixel784'), mas la columna objetivo **label**.
* **Volumen de datos:**
    * **Entrenamiento:** $27.455$ muestras.
    * **Prueba:** $7.172$ muestras.
    * **Total:** $34.627$ muestras.
* **Distribución de clases:**
    * Contiene 24 clases numéricas (valores de $0$ a $23$). El conjunto de datos viene con la exclusión de la letra **J** y la **Z**, ya que ambas señas requieren un gesto con movimiento, por lo que no puede representarse en una imagen estática.
    * **Variante del grupo:** Conjunto de 24 clases estáticas del 0 al 23.
* **Trazabilidad:**
   * **Entrenamiento:** 80% del dataset train ($21.964$ imágenes).
   * **Validación:** 20% del dataset train ($5.491$ imágenes).
   * **Prueba (Test final):** $7.172$ imágenes.

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
	Construcción de la red neuronal multicapa (MLP) en Tensor Flow, definiendo capas densas, funciones de activación ('ReLU', 'Softmax'), regularización y optimizador 'Adam'.
* **Evaluación:**
Análisis del rendimiento del modelo mediante los KPIs definidos.
* **Despliegue:**
Documentación técnica “**README.md**”, versionamiento de código y entrega de la solución reproducible.

## 6. Estructura del repositorio
```text
Sign-Language-MNIST/
├── images/                    # Imágenes del conjunto de datos
├── notebooks/                 # Notebook 'neuronaNetwork.ipynb'
├── .gitignore                 # Archivos para ignorar al clonar repositorio
├── .python-version            # Versión necesaria de python
└── README.md                  # Informe técnico en formato Markdown
```

## 7. Análisis Exploratorio EDA
### Librerías utilizadas

> Se utilizaron las siguientes librerias como refuerzo para este analisis:

- **pandas:** Exploracion de estructuras, nulos, tipos, estadistica.
- **numpy:** Analizar los valores y operaciones numericas.
- **matplotlib, seaborn :** Para una visualizacion exploratoria como distribuciones y relaciones.
- De las cuales se utilizaron `shape`, `describe`, `info`, `unique`, etc.
- **tensorflow:** Modelado de la red neuronal y construcción del dataset en tensores.

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
- Se formatearon las clases de 784 elementos a estructuras de matriz $(28, 28, 1)$ para la correcta visualización e inspección de las imágenes de señas.
- En la etapa de normalización se utiliza la función normalizar. En esta fase ocurren dos procesos: primero se realiza el casteo de los valores numéricos, convirtiendo los datos a float32, lo que permite trabajar con ellos como tensores de TensorFlow. Posteriormente, cada valor de intensidad de los píxeles se divide por 255, que corresponde al valor máximo de intensidad, obteniendo valores dentro del rango de 0 a 1.

## 8. Diseño e implementación MLP
| Capa | Tipo | Configuración |
| :--- | :---- | :--- |
| Entrada | Flatten | Aplana imagen a 784 pixeles (28, 28, 1) |
| Capa oculta 1 | Dense | 128 neuronas |
| Capa oculta 2 | Dense | 64 neuronas |
| Salida | Dense | 24 neuronas |

* **Hiperparámetros**
  * **Optimizador:** Adam con una tasa de aprendizaje de 'learning_rate=0.001'.
  * **Tamaño batch:** 32 imágenes.
  * **Épocas:** Se inicio con 5 épocas.

## 9. Limitaciones de MLP en imágenes
> Durante el desarrolllo se identificaron las siguientes limitaciones de MLP:
* **Escalabilidad:** Cuando se requiera imágenes a color (RGB) o de mayor resolución, los pesos en las capas densas aumentaría provocando un sobreajuste o llamada 'Overfitting'.
* **Perdida de estructura:** La capa **Flatten** destruye la relación de vecindad 2D entre pixeles.
* **Propuesta de mejora:** En un futuro se recomendaría implementar CNN (Redes Neuronales Convolucionales), para conservar la geometría.
