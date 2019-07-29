# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 02:42:08 2018

@author: AYSE
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#verilerin yukleme
veriler= pd.read_csv("eksikveriler.csv")

print(veriler)

#veri on isleme
boy= veriler[["boy"]]
print(boy)
cinsiyet=veriler[["cinsiyet"]]
print(cinsiyet)

boykilo=veriler[["boy","kilo"]]
print(boykilo)

x=10
class insan:
    boy=130
    def kosmak(self,b):
        return b+10
    
ali= insan()
print(ali.boy)
print(ali.kosmak(90))

#eksik veriler

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN', strategy='mean', axis=0)
Yas=veriler.iloc[:,1:4].values
print(Yas)
imputer=imputer.fit(Yas[:,1:4])
Yas[:,1:4]=imputer.transform(Yas[:,1:4])
print(Yas)





