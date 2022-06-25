# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 11:28:24 2021

@author: dougl
"""

import pandas as pd

colunas = "age,workclass,final-weight,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loos,hour-per-week,native-country,income".split(',')
base = pd.read_csv('adult.csv')

previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values
                
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

column_tranformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])],remainder='passthrough')
previsores = column_tranformer.fit_transform(previsores).toarray()

# labelencoder_classe = LabelEncoder()
# classe = labelencoder_classe.fit_transform(classe)

# Escalonamento dos dados numericos para a mesma escala.
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0)


from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores_treinamento, classe_treinamento)
# Testa a base de treinamentos com os valores da variavel previsores_teste, é esperado que o resultado obtido seja igual a classe_teste
previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
# Irá retornar qual é a taxa de acerto obtida com esta base de treinamento x base de teste
precisao = accuracy_score(classe_teste, previsoes)
# Irá retornar a quantidades de cada classe que foram classificadas corretamente e incorretamente. 
# As classes corretas serão as que i e j (linha e coluna) forem iguais.
matriz = confusion_matrix(classe_teste, previsoes)
