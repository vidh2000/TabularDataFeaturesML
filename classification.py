from preprocessing import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = loadData()

keys = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
       'petal width (cm)', 'species', 'species_id']

X = data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
       'petal width (cm)']]
y = data["species_id"]
# Assuming you have your data (X) and labels (y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=312)

# Get the example for later testing 
n=30

# RF
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy (RF):", accuracy)

#Try for a simple known example and see probabilities for classes
probs = clf.predict_proba(X_test.iloc[[n]])
y_pred = clf.predict(X_test.iloc[[n]])
print("Validation example class =", y_test.iloc[n])
print("Probabilities:\n", probs)
print("Prediction = ", y_pred)

# SVM
from sklearn import svm
clf = svm.SVC()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy (SVM):", accuracy)
