import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

model =  tf.keras.models.load_model("Mental_model_new_hdf5.h5")

Mental = pd.read_csv("https://raw.githubusercontent.com/kuchbhi-kunal/nidan/main/survey%20(1).csv")
Mental.drop("comments",axis=1,inplace=True)
Mental.drop("Timestamp",axis=1,inplace=True)
Mental.drop("state",axis=1,inplace=True)
Drop_list = [584, 613, 1159, 1209,11, 209, 680, 1179,139, 204, 526,418, 478, 1247,97, 180, 869,93, 192, 670,819, 821,1069, 1229,277, 1168,729, 1110,37, 281,1140,1174,989,1178,753,1208,750,334,655,639,532,523,421,409,390,319,133,129,1213]
Mental.drop(Drop_list,inplace=True)
Mental['self_employed'].fillna("No",inplace=True)
Mental['work_interfere'].fillna('Sometimes',inplace=True)

X= Mental.drop('mental_health_consequence',axis=1)
Y=Mental['mental_health_consequence']
X_train, X_test, Y_train1, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)

Columns = list(X_train.columns)
Columns.remove('Age')

ct = make_column_transformer(
    (MinMaxScaler(), ['Age']),
    (OneHotEncoder(handle_unknown='ignore'),Columns)) 

ct.fit(X_train)

def predict(Age,Gender,Country,self_employed,family_history,treatment,work_interfere,no_employees,remote_work,tech_company,benefits,care_options,wellness_program,seek_help,anonymity,leave,phys_health_consequence,coworkers,supervisor,mental_health_interview,phys_health_interview,mental_vs_physical,obs_consequence,Model=model,ct=ct):
 diagnose = {
 'Age':[Age],
 'Gender':[Gender],
 'Country':[Country],
 'self_employed':[self_employed],
 'family_history':[family_history],
 'treatment':[treatment],
 'work_interfere':[work_interfere],
 'no_employees':[no_employees],
 'remote_work':[remote_work],
 'tech_company':[tech_company],
 'benefits':[benefits],
 'care_options':[care_options],
 'wellness_program':[wellness_program],
 'seek_help':[seek_help],
 'anonymity':[anonymity],
 'leave':[leave],
 'phys_health_consequence':[phys_health_consequence],
 'coworkers':[coworkers],
 'supervisor':[supervisor],
 'mental_health_interview':[mental_health_interview],
 'phys_health_interview':[phys_health_interview],
 'mental_vs_physical':[mental_vs_physical],
 'obs_consequence':[obs_consequence]}

 df = pd.DataFrame(diagnose)
 df_ct = ct.transform(df)

 Y_preds = Model.predict(df_ct)
 y_pred = Y_preds.argmax(axis=1)

 return y_pred
