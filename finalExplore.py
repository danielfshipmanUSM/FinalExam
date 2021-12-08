"""
Daniel Shipman

Final Exam data exploration

12/4/2021

The goal of this is to get a sense of the data sets provided and try and develop an approach to to analyzing the datasets.

Important questions 

What are the independent variables?
What are the dependent variables?

Whats the population?

Are the samples independent? 

Are the samples identically distributed?

Are we dealing with little n large p?

The three datasets we're analyzing are Zoo, Pollution, and Land Cover

Zoo/zoo.csv
Pollution/Pollution.txt

Land Cover/testing.csv
Land Cover/training.csv
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model as lin
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import median_absolute_error

#this is for reproducibility on different machines, and for different datasets
path = "C:\\Users\\dfshi\\Documents\\School\\MAT496\\Final\\Data\\"
DATASET_FOLDER = "Zoo"
os.chdir(path + DATASET_FOLDER)
DATASET_NAME = "zoo.csv"

data = pd.read_csv(DATASET_NAME)

"""
For Zoology dataset,

Indpendent vars - everything except for type
Dependent var - class type

"""
p = len(data.columns)
X = data.iloc[:,1:p-1]
y = data.iloc[:,p-1]

#random state is set for reproducibility
#split the dataset for data validation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

# Try a Ridge regression classifier that cross validates and selects the best regularization param out of set alphas
clf = lin.RidgeCV(alphas=[1e-5,1e-4,1e-3, 1e-2, 1e-1, 1,1e2]).fit(X_train, y_train)
clf.score(X_test, y_test)
clf.get_params()

clf.alpha_

#try lasso regression
clf = lin.LassoCV(alphas=[1e-5,1e-4,1e-3, 1e-2, 1e-1, 1,1e2]).fit(X_train, y_train)
clf.score(X_test, y_test)
clf.get_params()
betas = clf.coef_

np.argmax(betas)

np.argsort(betas)[:5:-1]
np.argsort(betas)[:10]
zeros = []
for i in enumerate(betas):
     if i[1] == 0:
         zeros.append(i[0])
         
cut_vars = X.iloc[:,zeros]
clf.alpha_
dir(clf)
"""
both perform equally well with the same lambda = 0.1 

however in order 
"""

y_pred = clf.predict(X_train)
mae = median_absolute_error(y_train, y_pred)
string_score = f"MAE on training set: {mae:.2f} deviation from class"
y_pred = clf.predict(X_test)
mae = median_absolute_error(y_test, y_pred)
string_score += f"\nMAE on testing set: {mae:.2f} deviation from class"
fig, ax = plt.subplots(figsize=(5, 5))
plt.scatter(y_test, y_pred)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c="red")
plt.text(1, 8, string_score)
plt.title("Ridge model, small regularization")
plt.ylabel("Model predictions")
plt.xlabel("Truths")
plt.xlim([0, 10])
_ = plt.ylim([0, 10])

