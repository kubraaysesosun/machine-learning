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


#kategorik verileri sayısala cevirme

ulke= veriler.iloc[:,0:1].values
print(ulke)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
ulke[:,0] =le.fit_transform(ulke[:,0])
print(ulke)
#from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(categorical_features='all')
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

#verilerin birleştirilmesi ve dataframe oluşturulması
print(list(range(22)))
sonuc = pd.DataFrame(data=ulke, index=range(22), columns=['fr','tr','us'])  
print(sonuc)
sonuc2 = pd.DataFrame(data=Yas, index=range(22), columns=['boy','kilo','yas'])  
print(sonuc2)

cinsiyet=veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3=pd.DataFrame(data=cinsiyet,index=range(22),columns=['cinsiyet'])
print(sonuc3)

s=pd.concat([sonuc,sonuc2],axis=1)
print(s)
s2=pd.concat([s,sonuc3],axis=1)
print(s2)

#veri kumesinin egitim ve test olarak ayrilmasi
from sklearn.cross_validation import train_test_split #farklı bolme islemleride var
x_train, x_test,y_train,y_test=train_test_split(s,sonuc3,test_size=0.33,random_state=0)

#öznitelik ölcekleme
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)# x_train kumesini al buna buna standardscaler'ı uygula ve yeni uygulanmis halini X_train kumesi olarak yap
X_test=sc.fit_transform(x_test)



