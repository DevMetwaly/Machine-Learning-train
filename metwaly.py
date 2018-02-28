# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 21:57:12 2018

@author: ABDELRHMAN HAMDY METWALY
"""

import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor


label=[]

for line in open("abalone/abalone.domain","r").readlines():
    line =line.split(":")
    label.append(line[0])
    
df = pd.read_csv(r"abalone/abalone.data",names=label)

le = LabelEncoder()
df.sex = le.fit_transform(df.sex)


X=df.iloc[:,:df.shape[1]-1]
Y=df.iloc[:,df.shape[1]-1]

def fn(X,Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=1) 
    linReg=LinearRegression()
    linReg.fit(x_train,y_train)
    y_predict=linReg.predict(x_test)
    print(np.sqrt(mean_squared_error(y_test,y_predict)))

def fn2(X,Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=1) 
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf.fit(x_train, y_train)
    y_predict=svr_rbf.predict(x_test)
    print(np.sqrt(mean_squared_error(y_test,y_predict)))

def fn3(X,Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=1) 
    regr_1  =  DecisionTreeRegressor(max_depth=1)
    regr_1.fit(x_train, y_train)
    y_predict=regr_1.predict(x_test)
    print(np.sqrt(mean_squared_error(y_test,y_predict)))
    
def fn4(X,Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=1) 
    model = make_pipeline(PolynomialFeatures(5), Ridge())
    model.fit(x_train, y_train)
    y_predict=model.predict(x_test)
    print(np.sqrt(mean_squared_error(y_test,y_predict)))
    
def fn5(X,Y):
   
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.32, random_state=48) 
    XGB = XGBRegressor()
    XGB.fit(x_train, y_train)
    y_predict=XGB.predict(x_test)
    
    print(np.sqrt(mean_squared_error(y_test,y_predict)))

            
fn(X,Y)
#fn2(X,Y)
#fn3(X,Y)
#fn4(X,Y)
fn5(X,Y)