# -*- coding: utf-8 -*-
"""
#### ---------------------------------- ####
Created on Tue Mar  1 12:20:52 2022
@Author:  @BIVentures of @BIAnalyticsVentures
Purpose:Fraud Detection on Bank Financial Payment System
Data Transformation
Exploratory Data Analysis
Data Preprocessing/Feature Engineering
Classification
Data Source: Opensource Bank Payment Data (Banksim dataset)
#### ---------------------------------- ####
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
