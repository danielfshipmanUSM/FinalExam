"""
Daniel Shipman

Final Exam data exploration

12/4/2021

The goal of this is to get a sense of the data sets provided and try and develop an approach to to analyzing the datasets.

Important questions 

What are the independent variables?
What are the dependent variables?

Whats the population?


Are we dealing with little n large p?

The three datasets we're analyzing are Zoo, Pollution, and Land Cover

In this project we'll be focusing on pollution
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
import statsmodels.api as sm

#this is for reproducibility on different machines, and for different datasets
path = "/home/daniels/Documents/School/MAT496/FinalExam/Data/Data/"
DATASET_FOLDER = "Pollution"
os.chdir(path + DATASET_FOLDER)
DATASET_NAME = "Pollution.txt"

with open('columns.txt') as f:
    column_names = f.read().strip().split(' ')
data = pd.read_csv(DATASET_NAME, delimiter='\s+', index_col=False,header=None)
data.set_axis(column_names,axis=1, inplace=True)

"""
An interesting problem might be to see whether the mortality rate is effected at all by pollution.
    - Since Mortality rate seems to be normally distributed with no outliers, this probably isn't the question to ask
    - What about poverty? Does being poor indicate more exposure to pollution? 
    - 

Couple questions, can we determine the distribution of either the MORT HC NOX or SO variables?
    - Mortality rate seems to be pretty definitively normal, with few to none notable outliers
    -All the polution data is not normal however. Might be a good candidate for a bootstrap hypothesis test. 

"""
def checkNormVis(colname):
    col_data = data[colname]
    
    mean = np.mean(col_data)
    median = np.median(col_data)
    plt.axvline(mean, color = 'red', label='mean')
    plt.axvline(median, color = 'green', label='median')
    plt.hist(col_data)
    
    #QQplot to determine if normal, must standardize the data first
    col_data -=np.mean(col_data)
    col_data /=np.std(col_data)
    
    sm.qqplot(col_data, line='45')

checkNormVis('MORT')
#This is a plot of theoretical normal values verus the Mortality data, Indicates Mort is normal
checkNormVis('SO')
#This is interesting, it does NOT look SO pollution comes from a normal dist
checkNormVis('HC')
#This is interesting, it does NOT look HC pollution comes from a normal dist
checkNormVis('NOX')
#Same as other pollutions, could be power but thats just a guess

"""
Hypothesis test that high levels of poverty correlate to higher levels of ambient living pollution

    -The intuition for this is fairly simple, living in a poorer neighborhood might lead to less serviced appliances or in less desireable areas. This could be due to pollution
    Focux on NOX pollution
    
    Hnull = mean(NOX) = mean(NOX_high_poor)
    
    define high_pov  > 0.8 quantile
    
    just by comparing null mean to sample mean, they are essentially identical. This indicates the poor wouldn't effect NOX much so I decided to look elsewhere
    (high_pov mean = -0.21547241438417064 , sample mean = -0.21547241438417059)
    
    what about high levels of HC pollution?
    
    Null hypothesis
    
    mean(NOX) = mean(NOX_high_HC)
    sample mean = -0.21547241438417059
    under null hyp = 0.9148508907018157
    
    this might be statistically significant, to determine if it is we can bootsstrap it
    since sample mean < null hyp mean compute the percentile and see if its less than 0.05
    
"""
poor = data['HC']
nox = data['NOX']
truemean = np.mean(nox)
pov_lim = np.quantile(poor, 0.8)
high_pov = []

for i, e in enumerate(poor):
    if e > pov_lim:
        high_pov.append(nox.iloc[i])
        
#BOOTSTAP

#calc null hyp mean
nullmean = np.mean(high_pov)

#Shift data by new mean
nox -= np.mean(nox)
nox += nullmean

# Resample from null hypothes-shifted data

B = 10000 
resamples = np.zeros(B)
resample_size = int(len(nox)/2)

for i in range(B):
    xstar = np.random.choice(nox, size=resample_size)
    resamples[i] = np.mean(xstar)

# compute the 0.05 percentile    
perc = np.percentile(resamples, 0.05)

if truemean < perc:
    print("Reject the null hypothesis")
    
"""
There it is! We reject the hypothesis that the mean of NOX with high HC is the same as the mean of NOX, this implies that higher HC influences NOX values. 
Note that we could've also used the central limit theorem to generate a confidence interval since the data is independent, the sample size is >30 and we're comparing means .
"""
