# Sign-Language-MNIST

## Analisis Exploratorio EDA

> En primera instancia se procede a aplicar una metodologia para analisis exploratio de datos aplicando **EDA**.

    - se utilizaron las siguientes librerias como refuerzo para este analisis;
        - pandas, exploracion de estructuras, nulos, tipos, estadistica.
        - numpy , analizar los valores y operaciones numericas.
        - matplotli, para una visualizacion exploratoria como distribuciones y relaciones.
        - de las cuales se utilizaron shape, describe , info , unique ,etc.

> > primera fase se analiza de la siguiente manera

- conocimiento del dataset
- conocer las dimensiones de ambos archivos train y test
- conocer las clases y su cantidad.
- revisar valores.
- entender features y label.

> > procesamiento

- separacion de datos, features y label.
- normalizacion de pixeles.

> > preparacion de 1 pipeline tensores a dataset

- tf.data.Dataset

> > entrenamiento
