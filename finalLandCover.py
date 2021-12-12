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

This one focuses on Land Cover, Here it seems like we're dealing with images with a fairly small dataset. Here feature selection seems important. 

Since the dataset is labeled, supervised methods might work well, however since its little n large p, we might have trouble with simple methods like logistic regression 

Basic idea of exploratory structure,

try fitting a logistic model to the dataset, probably with feature selection (Maybe Lasso?)

try something more complex like Multilayered perceptron 

finally, try an unsupervised method (I'm thinking SVD and PCA')

Exploratory analysis shows that y data labels are strings, we need them to be integers to perform regression on. 
asphalt   = 0
building  = 1
car       = 2
concrete  = 3
grass     = 4
pool      = 5
shadow    = 6
soil      = 7
tree      = 8

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
DATASET_FOLDER = "LandCover"
os.chdir(path + DATASET_FOLDER)
DATASET_NAME_TRAIN = "training.csv"
DATASET_NAME_TEST = "testing.csv"

#Data's already split so we just have to load it into memory
train = pd.read_csv(DATASET_NAME_TRAIN)
test = pd.read_csv(DATASET_NAME_TEST)


p = len(train.columns)
X_train = train.iloc[:,1:p]
y_train = train.iloc[:,0]

X_test = test.iloc[:,1:p]
y_test = test.iloc[:,0]

dep_names = np.unique(y_train)

# Creates a dictionary that maps the unique values of y to numerical values
dep_encoder = dict(zip(dep_names,range(len(dep_names))))

for i,e in enumerate(y_train):
    y_train.iloc[i] = dep_encoder[e]

for i,e in enumerate(y_test):
    y_test.iloc[i] = dep_encoder[e]

# Try a Lasso regression classifier that cross validates and selects the best regularization param out of set alphas
# Since we're dealing with many features and few samples, risk overfitting the data if we dont perform feature selection
clf = lin.LassoCV(alphas=[1,1e2,1e3,1e4,1e5]).fit(X_train, y_train)
clf.score(X_train, y_train)
betas = clf.coef_
opt_alpha = clf.alpha_


