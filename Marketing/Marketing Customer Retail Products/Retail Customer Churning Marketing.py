"""
File: # -*Consumer Life Cycle - Data Analysis, Modelling and Visualization  -*-
Created on Fri Dec 31 05:55:00 2021
@Author:  @BIVentures of @BIAnalyticsVentures
#-*- Data Analysis, Modelling & Visualization -*-
#-*- Features Analysis (Numerical & Categorical), Modelling & Visualization -*-
#File: Retail Products Marketing, Customer Churning
Disclaimer: The code is withheld to avoid copycats/stealing of my work
# -*- coding: utf-8 -*-
"""

import pickle
import os.path
import time
import warnings
warnings.filterwarnings('ignore')

##Load Libraries
import re
import pandas as pd
#import pandas.util.testing as tm
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, fbeta_score
from sklearn.metrics import auc, roc_curve, roc_auc_score, precision_recall_curve

import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
#%config InlineBackend.figure_formats = ['retina']
import seaborn as sns


##Get & Set current working directory path
#path = os.getcwd()
#print(path)
#print('\n')

#filepath=os.chdir('filepath')
#print(filepath)
#print('\n')

##Loaddataset into dataframe
customer_df=pd.read_csv('filename.csv')
# create a dataframe 
customer_df = pd.DataFrame(customer_df) 
  
# converting each value  
# of column to a string 
customer_df = pd.DataFrame(data=customer_df)
