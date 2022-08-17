# -*- coding: utf-8 -*-
"""
Purpose:Model to Predict the length-of-stay for each patient at the time of admission
Cases with higher Lengh of Stay (LOS) risk can have their treatment plan 
optimized to minimize LOS and lower the chance of getting a hospital-acquired conditions 
Prior knowledge of LOS can aid in logistics such as bed management/planning/allocation
Created on Sun Nov 21 09:45:37 2021
@Author: BIAnalyticsVentures/BIVentures
File(s): Sample Data files - publicly available datasets MIMIC-III DB 
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
import ast


from sklearn.preprocessing import StandardScaler
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate

#Model Building and Evaluation

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report,f1_score,plot_confusion_matrix,recall_score
from sklearn.model_selection import GridSearchCV
print('Setup Complete')


#### Identify & Change Working Directory ####
print('System Path: ', sys.executable)
print('\n\n')
os.chdir('C:/Users/windo/Desktop/PythonVentures/HospitalLOS/datafiles')
currentDirectory=os.getcwd()
print(currentDirectory)	
print('\n')	

"""
# Environment settings: 
# Only use for sample dataset, big data will take forever!
pd.set_option('display.max_column', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_seq_items', None)
pd.set_option('display.max_colwidth', 500)
pd.set_option('expand_frame_repr', True) """

###### Step 1: Load the Data for Admissions ######
admissions_df = pd.read_csv('admissions.csv',index_col= None, na_values='?')
diagnosis_ICD_df = pd.read_csv('diagnosis_ICD.csv',index_col= None, na_values='?')
patients_df = pd.read_csv('patients.csv',index_col= None, na_values='?')
icustays_df = pd.read_csv('icustays.csv',index_col= None, na_values='?')

icustays_df.groupby('first_careunit').median()
print(icustays_df['hadm_id'].nunique())

# Feature engineering for Intensive Care Unit (ICU) category
# Reduce ICU categories to just ICU or MICU
icustays_df['first_careunit']=icustays_df['first_careunit'].replace({'CCU': 'ICU', 'CSRU': 'ICU','SICU': 'SICU','MICU': 'MICU', 'TSICU': 'ICU'})

icustays_df['cat'] = icustays_df['first_careunit']
