#%%
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
#%%
# Loading the dataset 
data = pd.read_csv('Dataset/CCPP_data.csv')
print(data.head())
# %%
# Data Exploration 

## Display the first few rows of the dataset
print(data.head())
## Check for missing values 
print(data.info()) 
## Check for duplicates
print(data.duplicated().sum())

### Summary satistics
print(data.describe())
# %%
