# -*- coding: utf-8 -*-
"""

Nicolas Gachancipa
Neural Network - Binary Classification

"""
# Imports.
# -------------------- #
import tensorflow.keras as k
from keras.callbacks import ModelCheckpoint 
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Define datasets.
# -------------------- #
iris = r"iris.data"
bank_loan = r"credit_card_database.csv"
wine_quality = r"wine_quality.csv"
breast_cancer = r"tumor.csv"
heart_attack = r"heart_attack.csv"
datasets = [iris, bank_loan, wine_quality, breast_cancer, heart_attack]


# Functions.
# -------------------- #
def neural_network(X, y):
    model = k.Sequential([k.layers.Dense(units=256, input_shape=[X.shape[1]], activation='relu'),
                          k.layers.Dense(units=128, activation='relu'),
                          k.layers.Dense(units=64, activation='relu'),
                          k.layers.Dense(units=32, activation='relu'), 
                          k.layers.Dense(units=16, activation='relu'), 
                          k.layers.Dense(units=8, activation='relu'), 
                          k.layers.Dense(units=1,  activation='sigmoid')])
    checkpoint = ModelCheckpoint('best_weights.h5', 
                                 verbose=1, monitor='loss', 
                                 save_best_only=True, mode='auto')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs= 100, callbacks=[checkpoint], verbose=1)
    model.load_weights('best_weights.h5')
    return model


# Datasets.
accuracies = []
# -------------------- #
for dataset in datasets:
    
    # Print database.
    #print('\n\nDataset:', [n for n in globals() if globals()[n] is dataset][0])
    #print('---------------------')
    
    # Split X and Y.
    dataset = pd.read_csv(dataset)
    X = dataset.drop('class', axis=1)
    y =  dataset['class']
    X, y = np.array(X), np.expand_dims(np.array(y), axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = neural_network(X_train, y_train)
    loss, accuracy = model.evaluate(X_test, y_test)
    accuracies.append(accuracy)
    print('Loss: ', loss, 'Accuracy:', accuracy*100)
    
# Print neural network accuracy.
print('Neural Network Accuracy:')
print('------------------------')
for i, j in zip(datasets, accuracies):
    print([n for n in globals() if globals()[n] is i][0], ': ', j*100, '%')