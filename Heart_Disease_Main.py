import joblib
import pandas as pd

model = joblib.load("Heart_Disease-Model.joblib")

def Prediction(Age,cp,age,sex,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,Model=model):
    par = {
        "Age":[Age],
        "cp":[cp],
        "age":[age],
        "trestbps":[trestbps],
        "chol":[chol],
        "fbs":[fbs],
        "restecg":[restecg],
        "thalach":[thalach],
        'exang':[exang],
        'oldpeak':[oldpeak],
        'slope':[slope],
        'ca':[ca],
        'thal':[thal]
    }

    df = pd.DataFrame(par)
    res = Model.predict(df)
    return res