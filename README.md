# Optimum AI Lab hackathon 2024.
Energy Consumption project is a machine learning prediction model that is aimed at forecasting the future consumption of energy consumers.
# Installation guide:

Import the various libraries in your IDE after you must have installed the required depenedencies in the anaconda prompt using pip install "the library"
the following libraries to instaall: 

sklearn

xgboost

lightgbm

pickle

missingno

# import the various libraries

#importing the necessary libraries

import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime

import datetime as dt

#importing warnings to help in ignoring warning issues/ messages

import warnings

warnings.filterwarnings('ignore')

import missingno as msno

#importing statistical libraries

import scipy.stats as stats

import statsmodels.api as sm

from scipy.stats import norm

#import ML libraries

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error 

from sklearn.linear_model import LinearRegression, Lasso, Ridge 

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import BaggingRegressor,  ExtraTreesRegressor, RandomForestRegressor, StackingRegressor

from sklearn.model_selection import GridSearchCV

from xgboost import XGBRegressor

import lightgbm

from lightgbm import LGBMRegressor

import pickle

# Data Source 

The data set for this project was collected from kaggle: https://www.kaggle.com/code/mrsimple07/energy-consumption-eda-prediction/input
