# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 12:23:55 2020
@Author:  @BIVentures of @BIAnalyticsVentures
"""

import pickle
import os.path
import time
import warnings
warnings.filterwarnings('ignore')

##Load Libraries
##for conda-
#conda install plotly
#conda install -c conda-forge cufflinks-py

import plotly
import plotly.graph_objs as go 
from plotly.offline import init_notebook_mode,iplot,plot
init_notebook_mode(connected=True)

import pandas as pd

##Get & Set current working directory path
path = os.getcwd()
#print(path)
#print('\n')

filepath=os.chdir('filepath/filename')
print(filepath)
#print('\n')
