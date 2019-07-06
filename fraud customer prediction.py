# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 22:13:18 2019

@author: Shubham
"""
# using som and then ann for probiblites
#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[ : , : -1].values
y = dataset.iloc[:, -1].values

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X = sc.fit_transform(X)

#fitting som
from minisom import MiniSom
som = MiniSom(x=10, y=10,
              input_len=15, 
              sigma=1.0,
              learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

#visualising the som
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
marker = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0]+0.5,
         w[1]+0.5,
         marker[y[i]],
         markeredgecolor= colors[y[i]],
         markerfacecolor= 'None',
         markersize= 10,
         markeredgewidth= 2)
show()

#finding fraouds
mapping = som.win_map(X)
frauds = np.concatenate((mapping[(2,5)], 
                                 mapping[(4,4)], 
                                 mapping[(6,5)], 
                                 mapping[(8,3)], 
                                 mapping[(8,8)]), 
                        axis= 0)
frauds = sc.inverse_transform(frauds)


#ANN

#matrix of features
customer = dataset.iloc[:, 1:16].values

#creating dependent variable
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1
y_train = is_fraud

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(customer)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 5, epochs = 10)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_train)
y_pred = np.concatenate((dataset.iloc[:,0:1].values, y_pred), axis=1)
y_pred = y_pred[y_pred[:, 1].argsort()]