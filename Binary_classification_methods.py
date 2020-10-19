# -*- coding: utf-8 -*-
"""

Nicolas Gachancipa
Binary Classification Methods:
    DecisionTreeClassifier
    Rule-Based Classifier
    K-Neighbours Classifier
    Naive-Bayes
    Suport Vector Machines
    Adaboost (Ensemble methos)

"""
# Imports.
# -------------------- #
from sklearn.tree import DecisionTreeClassifier
import wittgenstein as lw
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score

# Define datasets.
# -------------------- #
iris = r"iris.data"
bank_loan = r"credit_card_database.csv"
wine_quality = r"wine_quality.csv"
breast_cancer = r"tumor.csv"
heart_attack = r"heart_attack.csv"
datasets = [iris, bank_loan, wine_quality, breast_cancer, heart_attack]


# Classifier.
# -------------------- #
def compute_scores(clf, X, y):
    scores = cross_val_score(clf, X, y, cv=10, scoring = 'accuracy')
    return scores.mean(), scores.std()


# Datasets.
# -------------------- #
for dataset in datasets:
    
    # Print database.
    dataset_name = [n for n in globals() if globals()[n] is dataset][0]
    print('Dataset: ', dataset_name)
    
    # Read dataset, and obtain X and y matrices.
    dataset = pd.read_csv(dataset)
    X = dataset.drop('class', axis=1)
    y =  dataset['class']
    print('Size: ', X.shape[0])
    print('-------------------')
    
    # Compute scores.
    m1, s1 = compute_scores(DecisionTreeClassifier(criterion="entropy", splitter='random'), X, y)
    m2, s2 = compute_scores(lw.RIPPER(), X, y) # lw.IREP()
    m3, s3 = compute_scores(KNeighborsClassifier(n_neighbors=5), X, y)
    m4, s4 = compute_scores(GaussianNB(), X, y)
    m5, s5 = compute_scores(svm.SVC(C = 1), X, y)
    m6, s6 = compute_scores(AdaBoostClassifier(n_estimators=100, random_state=0), X, y)
    ms, ss = [m1, m2, m3, m4, m5, m6], [s1, s2, s3, s4, s5, s6]
    
    # Print results.
    results = pd.DataFrame(data = {'CV score': [round(100*i, 2) for i in ms],
                            '+-2std': [round(2*100*i, 2) for i in ss]}, 
                           index = ['Decision-tree', 'Rule-based',
                                    'K-neighbours', 'Naive Bayes', 
                                    'Support Vector Machine', 'Adaboost'])
    print(results)