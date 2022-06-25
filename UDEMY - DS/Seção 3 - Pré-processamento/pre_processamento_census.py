# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 18:23:16 2020

@author: dougl
"""
import pandas as pd
colunas = "age,workclass,final-weight,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loos,hour-per-week,native-country,income".split(',')
base = pd.read_csv('adult.csv', names=colunas)
base.columns
# o primeir ':' é referente as linhas e o segundo, as colunas
previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14:15].values

# Transformação de dados
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_previsores = LabelEncoder()
# Transformando os valores nominais em valores numericos
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
type(previsores[0])
indices_alterados = [1]
for i in range(len(previsores[0])):
    if type(previsores[0][i]) != int:
        indices_alterados.append(i)
        previsores[:, i] = labelencoder_previsores.fit_transform(previsores[:, i])
#3,5,6,7,8,9,13

# Tratamento dos valores nominais que viraram numericos para "Dummys".
# Obs: Os valores que eram nominais e se tornaram numericos irão se tornar Dummys, que seria como uma tabela verdade, 0 False e 1 True
onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), indices_alterados)], remainder='passthrough')
previsores = onehotencoder.fit_transform(previsores).toarray()

# Escalonamento dos valores da base de dados censu
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# previsores = scaler.fit_transform(previsores)
