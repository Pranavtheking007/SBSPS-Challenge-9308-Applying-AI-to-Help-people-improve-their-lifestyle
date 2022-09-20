#from crypt import methods
import flask
from flask import Flask , render_template , url_for , request , jsonify , redirect 
import tensorflow
import jinja2
from datetime import datetime
import joblib
import DiabetesMain as dia
import Heart_Disease_Main as hrt
import Mental_Health_Main as mlh
import Stroke_Main as stm

app = Flask(__name__)
# for home page clear
@app.route("/")
def home():
   return render_template("index.html")
#for choice page
@app.route("/choice.html")
def diagnose():
    return render_template("choice.html")
#for diabeties form
@app.route('/dia_form.html')
def dispform():
    return render_template("dia_form.html")
#form k badd ka isko link krna h main prediction model se
@app.route('/dia_form.html', methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        
        bmi = request.form["BMI"]
        Income = request.form["Income"]
        PhysHlth = request.form["PhysHlth"]
        Age = request.form["Age"]
        GenHlth = request.form["GenHlth"]
        HighBP = request.form["HighBP"]
        HighChol = request.form["HighChol"]
        Smoker = request.form["Smoker"]
        Stroke = request.form["Stroke"]
        HeartDiseaseorAttack = request.form["HeartDisease"]
        PhysActivity = request.form["PhysActivity"]
        Veggies = request.form["Veggies"]
        HvyAlcoholConsump = request.form["HeavyAlcoholConsump"]
        DiffWalk = request.form["DiffWalk"]
        Sex = request.form["Sex"]

    #x = joblib.load("DiabetesMain.py")  
        X=dia.Predict_dia(bmi,Income,PhysHlth,Age,GenHlth,HighBP,HighChol,Smoker,Stroke,HeartDiseaseorAttack,PhysActivity,Veggies,HvyAlcoholConsump,DiffWalk,Sex)
        res = str(X)
        return res
    # return render_template("X" , result=res)

@app.route("/Stroke_Form.html")
def function():
    return render_template("Stroke_Form.html")

@app.route("/Stroke_Form.html", methods = ["POST","GET"])
def fun1():
    if request.method =="POST":
        gender = request.form.get("gender")
        age = request.form["age"]
        hypertension = request.form.get("hypertension")
        heart_disease = request.form.get("heart_disease")
        ever_married = request.form.get("ever_married")
        work_type = request.form.get("work_type")
        Residence_type = request.form.get("Residence_type")
        avg_glucose_level = request.form["avg_glucose_level"]
        bmi = request.form["bmi"]
        smoking_status = request.form.get("smoking_status")
        
        w = stm.Predictions(gender, age, hypertension, heart_disease, ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status)
        res1 = str(w)
        return res1
    return render_template("w",result=res1)

@app.route("/Mental_Form.html")
def mental():
    return render_template("Mental_Form.html")
@app.route("/Mental_Form.html" , methods = ['POST','GET'])
def mental1():
    print(request.form)
    if request.form =="POST":
        
        Age = request.form['Age']
        Gender =  request.form.get("Gender")
        Country = request.form.get("Country")
        self_employed = request.form.get("self_employed")
        family_history = request.form.get("family_history")
        treatment = request.form.get("treatment")
        work_interfere = request.form.get("work_interfere")
        no_employees = request.form.get("no_employees")
        remote_work = request.form.get("remote_work")
        tech_company = request.form.get('tech_company')
        benefits = request.form.get("benefits")
        care_options = request.form.get("care_options")
        wellness_program = request.form.get("wellness_program")
        seek_help = request.form.get("seek_help")
        anonymity = request.form.get("anonymity")
        leave = request.form.get("leave")
        phys_health_consequence = request.form.get("phys_health_consequence")
        coworkers = request.form.get("coworkers")
        supervisor = request.form.get("supervisor")
        mental_health_interview = request.form.get("mental_health_interview")
        phys_health_interview = request.form.get("phys_health_interview")
        mental_vs_physical = request.form.get("mental_vs_physical")
        obs_consequence = request.form.get("obs_consequence")

        Y = mlh.predict(Age,Gender,Country,self_employed,family_history,treatment,work_interfere,no_employees,remote_work,tech_company,benefits,care_options,wellness_program,seek_help,anonymity,leave,phys_health_consequence,coworkers,supervisor,mental_health_interview,phys_health_interview,mental_vs_physical,obs_consequence)
        res3 = str(Y)
        print("Hello")
        return render_template("<h1>{res3}</h1>",result=res3)
        
    return render_template("<h1>Hello world</h1>")

     

@app.route("/Heart_Form.html")
def fun():
    return render_template("Heart_Form.html")    

@app.route("/Heart_Form.html",methods = ["POST" , "GET"])
def heart():
    if request.form == "POST":
       
        cp = request.form["cp"]
        age = request.form["Age"]
        sex = request.form["Sex"] 
        trestbps = request.form["Resting_Blood_Pressure"]
        chol = request.form["Cholestoral"]
        fbs = request.form["Fasting Blood Sugar"]
        restecg = request.form["Resting ECG"]
        thalatc = request.form["Max Heart Rate"]
        exang = request.form["Exercise_induced_angina"]
        Oldpeak  = request.form["ST_Depression"]
        slope = request.form["Slope_of_peak_exercise"]
        ca= request.form["ca"]
        thal=request.form["thal"]
        z = hrt.Prediction(cp,age,sex,trestbps,sex,trestbps,chol,fbs,restecg,thalatc,exang,Oldpeak,slope,ca,thal)   
        res = z
    return render_template("z", result=res)




     

if __name__ == "__main__":
    app.run(debug = True)   


#for form access
#@app.route("/form.html")
#def form():
#   return render_template("form.html")
#after form filling and for giving result
#@app.route("/result") 
#def result():
#   return request("Diabetes.ipynb") 






















#html k files k reference proper ---------------------------------------------->done
#form k elements k name dena proper na------------------------------------------->baki
#<input type="radio" name="gender" id="dot-------------------------------------baki
#index.html k badd new page pr redirect result k liye 
#home page will redirect to choice.html ------------------------------------>done
#button will redirect to disease page -----------------------------------.done
#diab ka route usme form hoga fill krna and save it ------------>baki
#analysis k badd info pass hoga in form of dict -------------------->baki
# result on new page . ------------.baki
#



#vs code todo download. --------------->done
#html page banana with 4 options and shall go to form and bharne k badd diagnose button work ------------>done
#diagnose route new request.form form save in sql alchemy ---------->not required
#sqllite better abhi k liye.
#request.form= dictionary (json hota h)
# access krne k




