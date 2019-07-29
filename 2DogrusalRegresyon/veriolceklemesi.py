# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 02:42:08 2018

@author: AYSE
"""

#1.Kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme

#2.1 Veri yukleme
veriler= pd.read_csv("satislar.csv")


#veri on isleme
aylar= veriler[["Aylar"]]
#test
print(aylar)

satislar= veriler[["Satislar"]]
print(satislar)

satislar2=veriler.iloc[:,:1].values
print(satislar2)


#veri kumesinin egitim ve test icin bolunmesi
from sklearn.cross_validation import train_test_split #farklı bolme islemleride var
x_train, x_test,y_train,y_test=train_test_split(aylar,satislar,test_size=0.33,random_state=0)

#verilerin ölceklenmesi
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)# x_train kumesini al buna buna standardscaler'ı uygula ve yeni uygulanmis halini X_train kumesi olarak yap
X_test=sc.fit_transform(x_test)



