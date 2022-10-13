import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import joblib
import pickle

Fit_Bit = pd.read_csv("https://raw.githubusercontent.com/Pranavtheking007/IBM_FIT-BIT/main/Fit_Bit_modified%20(1).csv")
Fit_Bit.drop(['Fat'],axis=1,inplace=True)

model = pickle.load(open("y_kmean_pred.pkl",'rb'))

for label, content in Fit_Bit.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            #Fill missing numeric valuees with median
            Fit_Bit[label] = content.fillna(content.median())

Numeric = []
for label, content in Fit_Bit.items():
    if pd.api.types.is_numeric_dtype(content):
      Numeric.append(label)

ct = make_column_transformer(
    (MinMaxScaler(), Numeric))

ct.fit(Fit_Bit)

X = Fit_Bit[['TotalSteps', 'VeryActiveMinutes']]
X = X.iloc[:, [0, 1]].values
Y = Fit_Bit[['TotalDistance', 'VeryActiveDistance']]
Y = Y.iloc[:, [0, 1]].values
Z = Fit_Bit[['Value', 'Calories']]
Z = Z.iloc[:, [0, 1]].values

def tips_pred(TotalSteps,TotalDistance,VeryActiveDistance,VeryActiveMinutes,SedentaryMinutes,Calories,WeightKg,BMI,Value,AverageIntensity,TotalTimeInBed):
    