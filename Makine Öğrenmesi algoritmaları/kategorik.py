print("Hello ml ")

import pandas as pd #Verileri düzgün bir şekilde tutmak için bu kütüphaneyi kullanırız
import numpy as np  # Büyük sayılar venumeric işlemler için kullanılır 
import matplotlib.pyplot as plt # Çizimler için kullanılır.

#kodlar 
#veri yükleme
veriler = pd.read_csv('eksikveriler.csv')
print(veriler)

#veri ön işleme
boy=veriler[['boy']]
print(boy)

boykilo = veriler[['boy','kilo']]
print(boykilo)


class insan:
    boy = 180
    def kosmak(self,b):
        return b + 10
ali = insan()
print(ali.boy)
print(ali.kosmak(90))

l = [1,3,4] 


#eksik veriler

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy='mean')

Yas =  veriler.iloc[:,1:4].values
print(Yas)
imputer =imputer.fit(Yas[:,1:4])
Yas [:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)
