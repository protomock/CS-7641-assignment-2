# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:23:40 2017

@author: JTay
"""

import numpy as np

import sklearn.model_selection as ms
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
df = pd.read_csv('https://raw.githubusercontent.com/protomock/data/master/ks-projects-201612.csv', index_col=0)
ksProjectsX = df.drop(['usd pledged ', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15',  'Unnamed: 16'],1).copy().values
ksProjectsY = df['usd pledged '].copy().values

trgX, tstX, trgY, tstY = ms.train_test_split(ksProjectsX, ksProjectsY, test_size=0.3, random_state=0)     

# pipe = Pipeline([('Scale',StandardScaler()),
#                  ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
#                  ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
#                  ('Cull3',SelectFromModel(RandomForestClassifier(random_state=3),threshold='median')),
#                  ('Cull4',SelectFromModel(RandomForestClassifier(random_state=4),threshold='median')),])
# trgX = pipe.fit_transform(ksProjects_trgX, ksProjects_trgY)
trgY = np.atleast_2d(trgY).T
# tstX = pipe.transform(ksProjects_trgX)
tstY = np.atleast_2d(tstY).T
trgX, valX, trgY, valY = ms.train_test_split(trgX, trgY, test_size=0.2, random_state=1)     
tst = pd.DataFrame(np.hstack((tstX,tstY)))
trg = pd.DataFrame(np.hstack((trgX,trgY)))
val = pd.DataFrame(np.hstack((valX,valY)))
tst.to_csv('m_test.csv',index=False,header=False)
trg.to_csv('m_trg.csv',index=False,header=False)
val.to_csv('m_val.csv',index=False,header=False)