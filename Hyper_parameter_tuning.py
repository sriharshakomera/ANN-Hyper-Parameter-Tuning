# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 13:06:15 2019

@author: Sriharsha Komera
"""

# Hyper parameter tuning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
path='F:\\Krish\\\ANN\\Churn_Modelling.csv'
dataset=pd.read_csv(path)
X=dataset.iloc[:,3:13]
y=dataset.iloc[:,13]

#creating dummy variables
geography=pd.get_dummies(X['Geography'],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)

#Concatinating
X=pd.concat([X,geography,gender],axis=1)

#Dropping the existing column
X=X.drop(['Geography','Gender'],axis=1)

#splitting the data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

## Performing the hyperparameter Optimization
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, LeakyReLU, BatchNormalization, Dropout
from keras.activations import relu, sigmoid

def create_models(layers, activation):
    model=Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_dim=X_train.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
            
    model.add(Dense(units=1, kernel_initializer= 'glorot_uniform', activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model=KerasClassifier(build_fn=create_models,verbose=0)

layers=[[20],[40,20],[45,30,15]]
activations=['sigmoid','relu']
param_grid=dict(layers=layers,activation=activations,batch_size=[128,256],epochs=[30])
grid=GridSearchCV(estimator=model, param_grid=param_grid,cv=5)

## Model Best Result
print(grid_result.best_score_,grid_result.best_params_)

# Prediction
y_pred=grid.predict(X_test)
y_pred=(y_pred>0.5)

# Confusion Matrix, Accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score

cm=confusion_matrix(y_test,y_pred)s
score= accuracy_score(y_test,y_pred)



# 

