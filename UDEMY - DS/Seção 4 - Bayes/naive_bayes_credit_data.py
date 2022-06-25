# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 17:36:19 2021

@author: dougl
"""


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

base = pd.read_csv('credit_data.csv')
# procurar valores onde a idade é menor que 0
base.loc[base['age'] < 0]
# retorna quais os valores nulos para a idade
base.loc[pd.isnull(base['age'])]
# Exibe a media para a idade
base['age'].mean()
# Substitui os valores de idade onde sao menores que 0 pela media das idades.
base.loc[base.age < 0, 'age'] = base['age'][base.age > 0].mean()        # Forma do cara

# Previsores são os valores que serão utilizados como input para o algoritmo
previsores = base.iloc[:, 1:4].values
# Classe serão os valores resultado, utilizando os previsores como base.
classe = base.iloc[:, 4].values

# 
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

# Normalizando os valores das colunas.
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Irá separar as bases para treinamento e para testes.
previsores_treinamneto, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores_treinamneto, classe_treinamento)
# Testa a base de treinamentos com os valores da variavel previsores_teste, é esperado que o resultado obtido seja igual a classe_teste
previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
# Irá retornar qual é a taxa de acerto obtida com esta base de treinamento x base de teste
precisao = accuracy_score(classe_teste, previsoes)
# Irá retornar a quantidades de cada classe que foram classificadas corretamente e incorretamente. 
# As classes corretas serão as que i e j (linha e coluna) forem iguais.
matriz = confusion_matrix(classe_teste, previsoes)
