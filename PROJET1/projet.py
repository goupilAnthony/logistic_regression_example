# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 10:04:43 2019

@author: antho
"""
import pandas as pd
import numpy as np
import seaborn as sns
import math
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('projet1.csv')


#2/ Construire les variables dépendantes et indépendante
X = data.iloc[:,[0,1,2]]
Y = data.iloc[:,3]


#4/ Gérer les variables catégoriques
le = LabelEncoder()
X['Country'] = le.fit_transform(X['Country'])
#FR = 0  /  Sp = 2  /  GER = 1
Y = le.fit_transform(Y)
#No =0  Yes=1
enc = OneHotEncoder(categorical_features = [0])
X = enc.fit_transform(X).toarray()

X = pd.DataFrame(X)

#Transoformer en entiers une colonne
col=4
for i in range(0,len(X.iloc[:,col])):
    X.iloc[i,col] = math.ceil(X.iloc[i,col])

#3/ Gérer les données manquantes
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imp.fit_transform(X)


#5/ Diviser le dataset entre test set et training set
X_train, X_test, y_train ,y_test = train_test_split(X,Y,test_size=0.25)


X = preprocessing.scale(X)


from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_pred)

classifier.score(X_test,y_test) 























