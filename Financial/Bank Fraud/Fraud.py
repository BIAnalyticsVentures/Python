# -*- coding: utf-8 -*-
"""
#### ---------------------------------- ####
Created on Tue Mar  1 12:20:52 2022
@Author:  @BIVentures of @BIAnalyticsVentures
Also @author: BI Ventures
Purpose:Fraud Detection on Bank Financial Payment System
Data Transformation
Exploratory Data Analysis
Data Preprocessing/Feature Engineering
Classification
Data Source: Opensource Bank Payment Data (Banksim dataset)
#### ---------------------------------- ####
"""
"""
Fraudulent behavior can be seen across many different fields such as e-commerce, 
healthcare, payment and banking systems. 
Fraud is a billion-dollar business and it is increasing every year. 
For these type of problems ML comes for help and reduce the risk of frauds and 
the risk of business to lose money. With the combination of rules and machine learning, 
detection of the fraud would be more precise and confident

Fact: The PwC global economic crime survey of 2018 [1] found that half (49 percent) 
of the 7,200 companies they surveyed had experienced fraud of some kind.

Even if fraud seems to be scary for businesses it can be detected using intelligent systems
such as rules engines or machine learning. Most people here in Kaggle are 
familier with machine learning but for rule engines here is a quick information. 
A rules engine is a software system that executes one or more business rules in a 
runtime production environment. These rules are generally written by domain experts
for transferring the knowledge of the problem to the rules engine and from there to production.
Two rules examples for fraud detection would be limiting the number of transactions 
in a time period (velocity rules), denying the transactions which come from previously 
known fraudulent IP's and/or domains.

Rules are great for detecting some type of frauds but they can fire a lot of false positives
or false negatives in some cases because they have predefined threshold values. 
For example let's think of a rule for denying a transaction which has an amount 
that is bigger than 10000 dollars for a specific user. If this user is an experienced 
fraudster, he/she may be aware of the fact that the system would have a threshold 
and he/she can just make a transaction just below the threshold value (10000-1 dollars).

"""

#### Import Libraries ####
#Filter warnings
import warnings
warnings.filterwarnings('ignore')
import sys
import os
import numpy as np
import pandas as pd
import datetime
from os import path
from pandas import read_csv
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import imblearn
print(imblearn.__version__)
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot
from columnar import columnar
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from numpy import where
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier


#### ---------------------------------- ####



#### Identify & Change Working Directory ####
print('System Path: ', sys.executable)
print('\n\n')
file_path ='filepath/filename'
os.chdir(file_path)
currentDirectory=os.getcwd()
print(currentDirectory)	
print('\n')	

#### ---------------------------------- ####

###### Step 1: Load the Data ######
