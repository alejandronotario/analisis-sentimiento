#!/usr/bin/env python

#title               :sentiment_analysis
#description         :Script para entrenar y predecir sentimientos de revisiones de Rotten Tomatoes
#author              :Alejandro Notario
#date                :2019-04-24
#version             :0.1
#usage               :sentiment_analysis.py
#requirements        :ficheros de entrenamiento y test proporcionados
#notes               :
#python version      :3.6
#==================================================================================

#librerías

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


train_data = pd.read_csv('./train.tsv',sep='\t')
test_data = pd.read_csv('./test.tsv',sep='\t')




def text_cleaning():
    def text_process(phrase):

        '''
        Esta función procesa el texto
        eliminando puntuación, tokenizando,
        y lematizando
        '''
        phrase = re.sub('[^a-zA-Z]|[0-9]', ' ',phrase)
        phrase = word_tokenize(phrase.lower())
        lemmatizer = WordNetLemmatizer()
        phrase = [lemmatizer.lemmatize(w) for w in phrase if not w in set(stopwords.words('english'))]
        return (' '.join(phrase))
    train_data['clean_phrases'] = train_data['Phrase'].apply(text_process)
    test_data['clean_phrases'] = test_data['Phrase'].apply(text_process)

    return train_data, test_data

def vectorizer():
    '''
    Función para vectorizar el contenido
    con el conteo de palabras y devuelve
    las revisiones ya procesadas y la columna de
    target del dataset de train para entrenar
    el modelo
    '''
    train_data, test_data = text_cleaning()
    cv = CountVectorizer(max_features = 1500, min_df=10)
    x__train = cv.fit_transform(train_data.clean_phrases).toarray()
    x__test= cv.fit_transform(test_data.clean_phrases).toarray()
    y = train_data.Sentiment.values
    print(x__train)
    return x__train, x__test, y

def split():
    '''
    devuelve los dataset de train y test
    '''
    x__train, x__test, y =vectorizer()
    X_train, X_test, y_train, y_test = train_test_split(x__train, y, test_size = 0.30,random_state = 0)
    print(y_test)
    return X_train, X_test, y_train, y_test

def log_reg():
    '''
    devuelve output de informe del modelo y
    fichero joblib con el modelo para cargarlo
    '''
    x__train, x__test, y =vectorizer()
    X_train, X_test, y_train, y_test = split()
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    pred_logReg = logreg.predict(x__test) #predicciones sobre el dataset de test
    print('Precisión del clasificador por Regresión Logística: {:.2f}'.format(logreg.score(X_test, y_test)))
    global confusion_matrix #this is to avoid unbound error
    cfm = confusion_matrix(y_test, y_pred)
    print("===============")
    print("matriz de confusión")
    print(cfm)
    print("===============")
    print("informe del modelo")
    print(classification_report(y_test, y_pred))
    return dump(logreg, 'logregmodel.joblib')

def naive_bayes():
    x__train, x__test, y =vectorizer()
    X_train, X_test, y_train, y_test = split()
    naive_bayes = MultinomialNB()
    naive_bayes.fit(X_train, y_train)
    y_pred = naive_bayes.predict(X_test)
    pred_naiveBayes = naive_bayes.predict(x__test)
    print('Precisión del clasificador por Naive Bayes: {:.2f}'.format(naive_bayes.score(X_test, y_test)))
    global confusion_matrix #this is to avoid unbound error
    cfm = confusion_matrix(y_test, y_pred)
    print("===============")
    print("matriz de confusión")
    print(cfm)
    print("===============")
    print("informe del modelo")
    print(classification_report(y_test, y_pred))
    return dump(naive_bayes, 'naiveBayes.joblib')

def svm():
    x__train, x__test, y =vectorizer()
    X_train, X_test, y_train, y_test = split()
    svm=SGDClassifier()
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    pred_svm = svm.predict(x__test)
    print('Precisión del clasificador por Máquina de Vectores de Soporte: {:.2f}'.format(svm.score(X_test, y_test)))
    global confusion_matrix #this is to avoid unbound error
    cfm = confusion_matrix(y_test, y_pred)
    print("===============")
    print("matriz de confusión")
    print(cfm)
    print("===============")
    print("informe del modelo")
    print(classification_report(y_test, y_pred))
    return dump(svm, 'svm.joblib')

def transfer_file():
    '''
    exporta fichero en .csv con
    columnas añadidas de las predicciones
    de cada uno de los modelos
    '''
    x__train, x__test, y =vectorizer()
    pipe1=load('./logregmodel.joblib')
    pipe2=load('./naiveBayes.joblib')
    pipe3=load('./svm.joblib')
    predregLog = pd.Series(pipe1.predict(x__test))
    pred_naiveBayes = pd.Series(pipe2.predict(x__test))
    pred_svm = pd.Series(pipe3.predict(x__test))
    test_data['Sentiment_logReg']=predregLog
    test_data['Sentiment_naiveBayes']=pred_naiveBayes
    test_data['Sentiment_svm']=pred_svm
    file=test_data.to_csv('result.csv')
    return file




def main():
    text_cleaning()
    vectorizer()
    split()
    log_reg()
    naive_bayes()
    svm()
    transfer_file()

if __name__ == '__main__':
    main()
