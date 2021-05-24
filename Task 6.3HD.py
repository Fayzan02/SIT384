# -*- coding: utf-8 -*-
"""
Created on Sat May  8 22:14:45 2021

@author: FAYZAN
"""

import wget
import numpy as np
import csv
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd
from sklearn import linear_model
import mglearn
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.svm import SVC # "Support Vector Classifier"






link_to_data ='https://raw.githubusercontent.com/Fayzan02/SIT384/main/spambase.data'
DataSet = wget.download(link_to_data)
df = pd.read_csv('spambase.data', header=None)


# The first 57 columns are features
# The last column has the correct labels (targets)
X, y = df.iloc[:, :57].values, df.iloc[:, 57].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("Logistic Regression")
logreg = LogisticRegression(max_iter=5000).fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(logreg.score(X_train, y_train)))
print("Accuracy on test set:{:.3f}".format(logreg.score(X_test, y_test)))

logreg100 = LogisticRegression(C=100, max_iter=5000).fit(X_train, y_train)
logreg10 = LogisticRegression(C=10, max_iter=5000).fit(X_train, y_train)



plt.plot(figsize=(7,7), dpi=100)
plt.plot(logreg.coef_.T, 'ko', label="C=1")
plt.plot(logreg100.coef_.T, 'r^', label="C=100")
plt.plot(logreg10.coef_.T, 'gv', label="C=10")
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.ylim(-8, 8)
plt.xlabel("Feature")
plt.ylabel("Coefficient magnitude")
plt.legend(loc = 3)
plt.show()

print() #Line Spacing


print("Random Forest Classifier")
forest = RandomForestClassifier(random_state=0).fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))

forest5 = RandomForestClassifier(n_estimators=5, random_state=0).fit(X_train, y_train)
forest10 = RandomForestClassifier(n_estimators=10, random_state=0).fit(X_train, y_train)

# fig, axes = plt.subplots(1, 3, figsize=(20, 5))
# for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):    
#     ax.set_title("Tree {}".format(i))    
#     mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)
    
# mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1], alpha=.4)
# axes[-1, -1].set_title("Random Forest")
# mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)


print() #Line Spacing

print("Support Vector Classifier")
svm = SVC(kernel='linear', C=1).fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(svm.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(svm.score(X_test, y_test)))

# mglearn.plots.plot_2d_separator(svm, X, eps=.5)
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# # plot support vectors
# sv = svm.support_vectors_
# # class labels of support vectors are given by the sign of the dual coefficients
# sv_labels = svm.dual_coef_.ravel() > 0
# mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")

                





