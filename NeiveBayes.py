# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 12:16:42 2019

@author: Rakin Shahriar
"""

import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest , chi2
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score
#load dataset weatherAUS.csv
data = pd.read_csv("weatherAUS.csv")
print("The data shape is ",data.shape)
#to view columns in ascending order of their valus. the columns those have minimun values will be rejected
t = data.count().sort_values() 
print(t)
#rejected columns with maximun null and minimum values
data = data.drop(columns=['RISK_MM','Sunshine','Evaporation','Cloud3pm','Cloud9am','Date','Location'])
print("New data shape after removing column",data.shape)
#for complete removal of na values
data = data.dropna(how='any')
print("Data shape without na = ",data.shape)
#z-score used to remove outliers.
#An outlier is a point that is significantly different from other points, which may causes serious error
z = np.abs(zscore(data._get_numeric_data()))
print(z)
data = data[(z<3).all(axis=1)]
print("Data shape after z-score = ",data.shape)
#replace yer and no to 1 and 0 respectively
data['RainToday'].replace({'No':0,'Yes':1},inplace = True)
data['RainTomorrow'].replace({'No':0,'Yes':1},inplace = True)
#See unique values and canvert them to integer using getDummies()
cetagorical = ['WindGustDir','WindDir3pm','WindDir9am']
for i in cetagorical:
    print(np.unique(data[i]))
#transform categaorcal columns
data = pd.get_dummies(data,columns=cetagorical)
#print(data.iloc[4:10])
#standardize values using mini max normalization
scale = MinMaxScaler()
scale.fit(data)
data = pd.DataFrame(scale.transform(data), index = data.index, columns = data.columns)
print(data.iloc[4:10])
x = data.loc[:,data.columns!='RainTomorrow']
y = data[['RainTomorrow']]
#print("View x columns ",x.columns)
select = SelectKBest(chi2,k=6)
select.fit(x,y)
x_new = select.transform(x)
print(x.columns[select.get_support(indices=True)])
data = data[['Rainfall','WindGustSpeed','Humidity9am','Humidity3pm','RainToday','WindDir9am_N','RainTomorrow']]
x = data.loc[:,data.columns!='RainTomorrow']
y = data['RainTomorrow']
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.25)
classifier = GaussianNB()
classifier.fit(x_train,y_train)
print(classifier.fit(x_train,y_train))
y_pred = classifier.predict(x_test)
conf_matrix = confusion_matrix(y_test,y_pred)
print("Confusion Matrix : ","\n",conf_matrix)
print("The accuracy score = ",accuracy_score(y_test,y_pred)*100)
conf_matrix
print("The precision score = ",precision_score(y_test,y_pred,average='micro')*100)
conf_matrix
print("The recall score = ",recall_score(y_test,y_pred,average='micro')*100)