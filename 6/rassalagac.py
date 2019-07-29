# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 02:42:08 2018

@author: AYSE
"""

#1.Kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score


# Veri yukleme
veriler= pd.read_csv("maaslar.csv")


#eğitimseviyes=x, maaslar=y olarak bölüyoruz
#dataframe slice
x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]
#numpy array dönüşümü
X=x.values
Y=y.values
#linear regression
#doğrusal model oluşturma
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)



#polynomial regression
#doğrusal olmayan(nonlinear) model oluşturma-2.derecen polinom
from sklearn.preprocessing import PolynomialFeatures

poly_reg=PolynomialFeatures(degree=2)
x_poly=poly_reg.fit_transform(X)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)

#4.dereceden polinom
poly_reg3=PolynomialFeatures(degree=4)
x_poly3=poly_reg3.fit_transform(X)
lin_reg3=LinearRegression()
lin_reg3.fit(x_poly3,y)



#Görselleştirme
plt.scatter(X,Y,color='red')
#linear regression gibi kullanıyoruz fakat predict ederken herhangi bir değer vermeden polinomal features a dönüştürmeliyiz
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.show()

#2 boyutlu uzayda x,y i dağıtma
plt.scatter(X,Y,color='red')
#x'in karşılığı olan lin_reg içinden predict etme
plt.plot(x,lin_reg.predict(X),color='blue')
plt.show()



plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg3.predict(poly_reg3.fit_transform(X)),color='blue')
plt.show()

#predicts

print(lin_reg.predict(11))
print(lin_reg.predict(6.6))

print(lin_reg2.predict(poly_reg.fit_transform(11)))
print(lin_reg2.predict(poly_reg.fit_transform(6.6)))




#verilerin ölceklenmesi
from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
x_scaler=sc1.fit_transform(X)# x_train kumesini al buna buna standardscaler'ı uygula ve yeni uygulanmis halini X_train kumesi olarak yap
sc2=StandardScaler()
y_scaler=sc2.fit_transform(Y)


#SVR Regresyon
from sklearn.svm import SVR
svr_reg=SVR(kernel='rbf')
svr_reg.fit(x_scaler,y_scaler)

plt.scatter(x_scaler,y_scaler,color='red')
plt.plot(x_scaler,svr_reg.predict(x_scaler),color='blue')
plt.show()
print(svr_reg.predict(11))
print(svr_reg.predict(6.6))




#Decision Tree Regresyon
from sklearn.tree import DecisionTreeRegressor
r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)#X'ten Y'yi öğren
Z=X+0.5
K=X-0.4
W=X+0.1
plt.scatter(X,Y,color='red')
plt.plot(x,r_dt.predict(X),color='blue')
plt.plot(x,r_dt.predict(Z),color='green')
plt.plot(x,r_dt.predict(K),color='yellow')
plt.plot(x,r_dt.predict(W),color='black')
plt.show()
print(r_dt.predict(11))
print(r_dt.predict(6.6))


#Random Forest Regresyonu
from sklearn.ensemble import RandomForestRegressor

rf_reg=RandomForestRegressor( n_estimators=10,random_state=0)
rf_reg.fit(X,Y)
print(rf_reg.predict(6.5))

plt.scatter(X,Y,color='red')
plt.plot(x,rf_reg.predict(X),color='blue')
plt.plot(x,rf_reg.predict(Z),color='yellow')
plt.plot(x,rf_reg.predict(K),color='green')
plt.show()

print("Random Forest R2 Değeri")
print(r2_score(Y,rf_reg.predict(X))) #gerçek Y(nin alacağı değerler) değerler actual değerler
print(r2_score(Y,rf_reg.predict(K)))
print(r2_score(Y,rf_reg.predict(Z)))


#summary R2 değerleri
print("*************")
print("Linear Regression R2 Değeri")
print(r2_score(Y,lin_reg.predict(X)))

print("Polynomial Regression R2 Değeri")
print(r2_score(Y,lin_reg2.predict(poly_reg.fit_transform(X))))

print("SVR R2 Değeri")
print(r2_score(y_scaler,svr_reg.predict(x_scaler)))

print("Decision Tree R2 Değeri")
print(r2_score(Y,r_dt.predict(X)))

print("Random Forest R2 Değeri")
print(r2_score(Y,rf_reg.predict(X)))
