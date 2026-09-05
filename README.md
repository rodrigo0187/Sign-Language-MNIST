# Sign-Language-MNIST

## Análisis Exploratorio EDA

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

- Separacion de datos, `features` y `label`.
- Normalizacion de pixeles.
-

### Preparación de 1 pipeline: tensores a Dataset

- En la etapa de normalización se utiliza la función normalizar. En esta fase ocurren dos procesos: primero se realiza el casteo de los valores numéricos, convirtiendo los datos a float32, lo que permite trabajar con ellos como tensores de TensorFlow. Posteriormente, cada valor de intensidad de los píxeles se divide por 255, que corresponde al valor máximo de intensidad, obteniendo valores dentro del rango de 0 a 1.

- Posteriormente, los tensores de características y sus etiquetas se utilizan para construir un tf.data.Dataset, manteniendo la correspondencia entre cada imagen y su etiqueta.

### Entrenamiento
