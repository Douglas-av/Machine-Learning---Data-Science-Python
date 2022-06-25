# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 17:14:11 2020

@author: dougl
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


base = pd.read_csv('credit_data.csv')
# Default = 0 (Pagou) | Default = 1 (Nao pagou)
base.columns = ['clientid', 'income', 'age', 'loan', 'default']
base.describe()
base.loc[base['age'] < 0]
# Apagar a coluna
# base.drop('age', 1, inplace=True)
# Apagar somente os registros com problema
# base.drop(base[base.age < 0].index, inplace=True)
# Preencher os valores manualmente
# Preencher os valores com a media
base.mean()
base['age'].mean()                                                      # Nesta media esta sendo considerada os valores negativos. Errada.
base['age'][base.age > 0].mean()                                        # Nesta media é descartado os valores negativos. Correta.
# base['age'][base[base.age < 0].index] = base['age'].mean()            # Minha forma
base.loc[base.age < 0, 'age'] = base['age'][base.age > 0].mean()        # Forma do cara
base['age'].describe()

base.isnull().sum()
pd.isnull(base['age'])                                                  # dificil a visualização dos valores nulos
base.loc[pd.isnull(base['age'])]                                        # Retonar as linhas onde contem valores nulos na coluna procurada
# base.loc[base['age'].isnull()]                                        # Retonar as linhas onde contem valores nulos na coluna procurada
base.loc[base['age'].isnull(), 'age'] = base['age'][base.age > 0].mean()
base['age'].fillna(base['age'][base.age > 0].mean(), inplace=True)
base.loc[base['clientid'].isin([29, 31, 32])]

base['default'].value_counts()
sns.countplot(x=base['default'],)
plt.hist(x=base['age'])
plt.hist(x=base['income'])
plt.hist(x=base['loan'])
grafico = px.scatter_matrix(base, dimensions=['age', 'income', 'loan'], color='default')         # grafico de dispersão (scatter)
grafico.show()


#----------------------------------------------------------------------------------------#
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

from sklearn.impute import SimpleImputer
# Estes já sao os parametros padrao da Imputer, só coloquei para saber quais são. CTRL+I para ajuda.
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])

from sklearn.preprocessing import StandardScaler
# Normalização e padronização dos valores fornecidos na base de dados.
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
#-----------------------------------------------------------------------------------------#
