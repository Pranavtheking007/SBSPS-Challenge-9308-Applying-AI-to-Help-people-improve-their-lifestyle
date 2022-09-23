import sqlite3


connection_dia = sqlite3.connect('dia_pred.db')
cursor_dia = connection_dia.cursor()
cursor_dia.execute("create table diadb (bmi real,Income real,PhysHlth real,Age integer,GenHlth real,HighBP integer,HighChol integer,Smoker integer,Stroke integer,HeartDiseaseorAttack integer,PhysActivity integer,Veggies integer,HvyAlcoholConsump integer,DiffWalk integer,Sex integer)")

connection_dia.commit()
connection_dia.close()
