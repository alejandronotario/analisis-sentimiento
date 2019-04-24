# sentiment_analysis


<br>
<hr>

## Resumen

En este repositorio se reúnen en un notebook 3 métodos de clasificación y predición de sentimientos según la crítica de múltiples usuarios a películas en el sitio we _Rotten Tomatoes_. El objetivo es predecir estos sentimientos en un dataset de test en el que no están etiquetadas estas revisiones. Para ello se dispone de un dataset para el entrenamiento de los modelos y otro dataset para aplicar los modelos, _train.tsv_ y _test.tsv_. En ellos se dispone del texto de las revisiones de los ususarios, que es lo que se utilizará como variable para el entrenamiento de los modelos. Los sentimientos son clasificados en 5 tipos, del 0 añ 4, negativo, algo negativo, neutral, algo positivo, positivo.

## Requerimientos

```python

import pandas as pd
import numpy as np
import re
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#NLTK 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

#Sklearn 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from joblib import dump, load
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

```

## Contenido del repositorio

- data:
	- train.tsv y test.tsv: ficheros para la realización del trabajo 
	- submission.csv: fichero de resultados 
	- result.csv: fichero en el que se incluyen las predicciones de los 3 modelos aplicados
- src:
	- sentiment_analysis.ipynb: notebook con el código explicando los pasos
	- sentiment_analysis.py: script para obtención de resultados de modelos y fichero con las predicciones de cada uno
	- Modelos:
		- logregmodel.joblib
		- naiveBayes.joblib
		- svmBayes.joblib


## Flujo de trabajo

- Exploración de los datos: Se comienza con una exploración para conocer el tipo de datos que tenemos así como dimensiones y distribución de tipo de sentimientos
- Procesamiento de texto: Es necesario procesar el texto para:
	- Eliminar puntación
	- Tokenizar, que divide el texto en piezas más pequeñas
	- Lematizar, que devuelve los lemas
	- Vectorización, que devuelve el vocabulario con sus ocurrencias 
- Preparación de los datos para estrenamiento y aplicación de modelos: consiste en dividie el fichero el contenido procedente del fichero de entrenamiento en dos partes una para entrenar X_train y otra para testear y evaluar X_test
- Entrenamiento de modelos: En este repositorio se hace uso de 3 modelos procedentes de las librerías de [scikitlearn](https://scikit-learn.org):
	- [Regresión Logística](https://en.wikipedia.org/wiki/Logistic_regression)
	- [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
	- [SVM (Máquina de vectores de soporte)](https://en.wikipedia.org/wiki/Support-vector_machine)
- Evaluación de los modelos. Para esto se estudia la matriz de confusión y un informe con precisión, recall, f1score:
	- Matriz de confusión: en las columnas figuran las predicciones de cada una de las clases y en las filas los valores reales, de forma que cuanto mayores sean los valores de la diagonal respecto al resto de campos, mejor es la precisión
	- Precisión: es la resultante de los verdaderos positivos(VP)/VP + falsos positivos (FP)
	- Recall: resutante de VP/VP+FN
	- f1score: media ponderada de precisión y recall
- Selección de modelo para aplicar sobre dataset de test. Se selecciona el que se considera más preciso y equilibrado para las distintas clases de sentimiento
- Guardar modelo
- Cargar modelo
- Aplicar modelo y exportar fichero Submission.csv

## Conclusiones

Hay margen de mejora en cualquiera de estos 3 modelos, para una primera entrega se ha seleccionado la Regresión Logistica. Como plan de trabajo, los siguientes pasos serían:

- Modificación de parámetro e hiperparámetros de los modelos para su mejora.
- Uso y desarrollo de otros modelos. Se podría aplicar la modelización de tópicos mediante [LDA](https://es.wikipedia.org/wiki/Latent_Dirichlet_Allocation) y el uso de librerías de Deep Learning  [Keras](https://keras.io/)
- Ir añadiendo más datos según se generen para poder entrenar los modelos con más volumen de datos.



