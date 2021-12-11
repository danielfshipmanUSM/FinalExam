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
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC

#this is for reproducibility on different machines, and for different datasets
path = "/home/daniels/Documents/School/MAT496/FinalExam/Data/Data/"
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
betas = clf.coef_
opt_alpha = clf.alpha_


np.argmax(betas)

coeffs_list = np.sort(betas)
features_ordered = X.columns[np.argsort(betas)]

#create dataframe for easy visualization of coefficients
coefs = pd.DataFrame(np.around(coeffs_list,3),columns=["Coefficients"], index=features_ordered)
print(coefs.info)

#Displays graph of performance of the model 
y_pred = clf.predict(X_train)
MSE = mean_squared_error(y_train, y_pred)
string_score = f"MSE on training set: {MSE:.2f} deviation from class"
y_pred = clf.predict(X_test)
MSE = mean_squared_error(y_test, y_pred)
string_score += f"\nMSE on testing set: {MSE:.2f} deviation from class"
fig, ax = plt.subplots(figsize=(5, 5))
plt.scatter(y_test, y_pred)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c="red")
plt.text(1, 8, string_score)
plt.title(f"Ridge model, alpha = {opt_alpha}")
plt.ylabel("Model predictions")
plt.xlabel("Truths")
plt.xlim([0, 10])
_ = plt.ylim([0, 10])


#try lasso regression
clf = lin.LassoCV(alphas=[1e-5,1e-4,1e-3, 1e-2, 1e-1, 1,1e2]).fit(X_train, y_train)
clf.score(X_test, y_test)
clf.get_params()
betas = clf.coef_
opt_alpha = clf.alpha_


np.argmax(betas)

coeffs_list = np.sort(betas)
features_ordered = X.columns[np.argsort(betas)]

#create dataframe for easy visualization of coefficients
coefs = pd.DataFrame(np.around(coeffs_list,3),columns=["Coefficients"], index=features_ordered)
print(coefs.info)


zeros = [i for i, e in enumerate(betas) if e == 0]
clf.alpha_
dir(clf)
"""
both perform equally well with the same lambda = 0.01 for Lasso and lambda =0.1 for Ridge

however in order 
"""

y_pred = clf.predict(X_train)
MSE = mean_squared_error(y_train, y_pred)
string_score = f"MSE on training set: {MSE:.2f} deviation from class"
y_pred = clf.predict(X_test)
MSE = mean_squared_error(y_test, y_pred)
string_score += f"\nMSE on testing set: {MSE:.2f} deviation from class"
fig, ax = plt.subplots(figsize=(5, 5))
plt.scatter(y_test, y_pred)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c="red")
plt.text(1, 8, string_score)
plt.title(f"Lasso model, alpha = {opt_alpha}")
plt.ylabel("Model predictions")
plt.xlabel("Truths")
plt.xlim([0, 10])
_ = plt.ylim([0, 10])

#try the dataset without the cut vars
cut_vars = X.columns[zeros]
X_cut = X.drop(columns=cut_vars)


c_list = np.arange(0.1,1,0.01)
kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']
c_score = np.zeros((len(kernel_list),len(c_list)))
                   
for i,k in enumerate(kernel_list):
    for j,c in enumerate(c_list):
        clf = SVC(C=c,kernel=k).fit(X_train,y_train)
        c_score[i][j] = clf.score(X_test,y_test)

kernel_opt = 'linear'
c_opt = np.round(c_list[np.argmax(c_score[0])], 3)
