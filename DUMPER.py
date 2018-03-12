# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:23:40 2017

@author: JTay
"""

import numpy as np
import pandas as pd
import re
import string
import math
from scipy.stats.mstats import winsorize 
import matplotlib.pyplot as plt

import sklearn.model_selection as ms
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectFromModel
df = pd.read_csv(
    'https://raw.githubusercontent.com/protomock/data/master/ks-projects-201612.csv', index_col=0)
df = df.drop(['Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15',  'Unnamed: 16'], 1)

# BAD VALUES
# Check for bad output values
condition = (df['usd pledged '].isnull()) | \
(df['usd pledged '] == 'US') | \
(df['usd pledged '] == 'CA') | \
(df['usd pledged '] == 'AU') | \
(df['usd pledged '] == 'GB') | \
(df['usd pledged '] == 'MX') | \
(df['usd pledged '] == 'SG') | \
(df['usd pledged '] == 'NZ') | \
(df['usd pledged '] == 'NL') | \
(df['usd pledged '] == 'DK') | \
(df['usd pledged '] == 'IE') | \
(df['usd pledged '] == 'NO') | \
(df['usd pledged '] == 'SE') | \
(df['usd pledged '] == 'DE') | \
(df['usd pledged '] == 'FR') | \
(df['usd pledged '] == 'ES') | \
(df['usd pledged '] == 'IT') | \
(df['usd pledged '] == 'AT') | \
(df['usd pledged '] == 'BE') | \
(df['usd pledged '] == 'CH') | \
(df['usd pledged '] == 'LU') | \
(df['usd pledged '] == 'HK') | \
(df['usd pledged '] == 'N,"0') | \
(df['usd pledged '] == 'failed') | \
(df['usd pledged '] == 'successful')

# Capture valid pledged values in usd pledged
df.loc[condition, 'usd pledged '] = df.loc[condition, 'pledged ']

# Set all usd pledged that came from pledged that were not valid to 0
df.loc[pd.to_numeric(df['usd pledged '], errors='coerce').isnull(), 'usd pledged '] = 0

# ENCODING
# Encode categorical values
for col in ['category ', 'main_category ', 'currency ', 'state ', 'country ']:
    df[col]=LabelEncoder().fit_transform(df[col])

# Convert the date to a vector of day, month, year
# Remove bad values, removing bad values for deadline removes bad launched values
df = df[~pd.to_datetime(df['deadline '], errors='coerce').isnull()]

# DATES
# Convert string datetime to actual datetime
df['launched '] = pd.to_datetime(df['launched '])
df['launched_year'] = df['launched '].dt.year

df['deadline '] = pd.to_datetime(df['deadline '])
df['deadline_year'] = df['deadline '].dt.year

# drop columns already represented
df = df.drop(['launched ', 'deadline '], 1)

# Capture character counts for each name
# for letter in list(string.ascii_lowercase):
#     df[letter] = df['name '].str.count(letter, re.IGNORECASE)

df = df.drop(['name '], 1)

# Check correlated columns
corr = df.corr()
fig, ax = plt.subplots(figsize=(12, 12))
ax.matshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()

# Drop correlated columns
df = df.drop(['deadline_year','pledged ', 'country '], 1)

# Convert column to numeric
df['usd pledged '] = pd.to_numeric(df['usd pledged '])

# Blend outliers
df['usd pledged '].plot.box(vert=False)
plt.show()
df['usd pledged '] = winsorize(df['usd pledged '], limits=[0.05, 0.05])
df['usd pledged '].plot.hist(bins=10)
plt.show()

# Set bins
df['bins'] = pd.cut(df['usd pledged '], bins=10)
df['bins'] = LabelEncoder().fit_transform(df['bins'])
df['bins'].plot.hist(bins=10)
plt.show()

# Drop the usd pledged
df = df.drop(['usd pledged '], 1)

dfX = df.drop(['bins'], 1).copy()
dfY=df['bins'].copy()

ksProjectsX=dfX.values
ksProjectsY=dfY.values

trgX, tstX, trgY, tstY=ms.train_test_split(
    ksProjectsX, ksProjectsY, test_size = 0.3, random_state = 0)

trgY=np.atleast_2d(trgY).T
tstY=np.atleast_2d(tstY).T
trgX, valX, trgY, valY=ms.train_test_split(
    trgX, trgY, test_size = 0.2, random_state = 1)

tst=pd.DataFrame(np.hstack((tstX, tstY)))
trg=pd.DataFrame(np.hstack((trgX, trgY)))
val=pd.DataFrame(np.hstack((valX, valY)))
tst.to_csv('m_test.csv', index = False, header = False)
trg.to_csv('m_trg.csv', index = False, header = False)
val.to_csv('m_val.csv', index = False, header = False)
