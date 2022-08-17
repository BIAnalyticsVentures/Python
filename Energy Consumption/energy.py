# -*- coding: utf-8 -*-
"""
File: Power Consumption Analytics
Created on Mon Feb 14 17:48:06 2022
@Author:  @BIVentures of @BIAnalyticsVentures
"""
##Load Libraries
import pandas as pd
import pickle
import string
import os.path
import time
import warnings
warnings.filterwarnings('ignore')
import plotly
import plotly.graph_objs as go 
import plotly.express as px
from plotly.offline import init_notebook_mode,iplot,plot
init_notebook_mode(connected=True)
from matplotlib import pyplot

##Get & Set current working directory path
path = os.getcwd()
#print(path)
#print('\n')

filepath=os.chdir('filename.csv')
print(filepath)
#print('\n')

powerconsump_df = pd.read_csv('filename.csv')
powerconsump_df.info()
print(powerconsump_df.head())

powerconsump_df.plot()

##Rest of Code withheld 
