from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import os
import scipy.stats as sp
import mlflow.sklearn
from sklearn.compose import make_column_transformer
import pandas as pd

os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://10.43.101.155:8081"
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'

mlflow.set_tracking_uri("http://10.43.101.155:8082/")
model = mlflow.sklearn.load_model("models:/random_forest@production")


class Diabetes(BaseModel):
    race: str = "Caucasian"                  
    gender: str = "Male"                  
    age: str = "[0-10)"                     
    weight: str = "[0-25)"                  
    admission_type_id: int = 6       
    discharge_disposition_id: int = 25
    admission_source_id: int = 7 
    time_in_hospital: int = 1
    payer_code: str = "BC"             
    medical_specialty: str = "Cardiology"
    num_lab_procedures: int = 41
    num_procedures: int = 0
    num_medications: int = 1
    number_outpatient: int = 0
    number_emergency: int = 0
    number_inpatient: int = 0
    diag_1: str = "250.83"                 
    diag_2: str = "250.01"                 
    diag_3: str = "255"
    number_diagnoses: int = 1
    max_glu_serum: str = ">200"          
    A1Cresult: str = ">8"              
    metformin: str = "No"              
    repaglinide: str = "No"            
    nateglinide: str = "No"            
    chlorpropamide: str = "No"         
    glimepiride: str = "No"            
    acetohexamide: str = "No"          
    glipizide: str = "No"              
    glyburide: str = "No"              
    tolbutamide: str = "No"            
    pioglitazone: str = "No"           
    rosiglitazone: str = "No"          
    acarbose: str = "No"               
    miglitol: str = "No"               
    troglitazone: str = "No"           
    tolazamide: str = "No"             
    examide: str = "No"                
    citoglipton: str = "No"            
    insulin: str = "No"                
    glyburide_metformin: str = "No"     
    glipizide_metformin: str = "No"    
    glimepiride_pioglitazone: str = "No"
    metformin_rosiglitazone: str = "No"
    metformin_pioglitazone: str = "No" 
    change: str = "No"                 
    diabetesMed: str = "No"            
    readmitted: str = "NO"             


    
app = FastAPI()

@app.get("/")
async def root():
    
    return {"message": "Diabetes Inference API"}

@app.post("/predict/")
def predict_readmitted_random_forest(diabetes: Diabetes):

    data = {
        "race": [diabetes.race],                  
        "gender": [diabetes.gender],                  
        "age": [diabetes.age],                     
        "weight": [diabetes.weight],                  
        "admission_type_id": [diabetes.admission_type_id],       
        "discharge_disposition_id": [diabetes.discharge_disposition_id],
        "admission_source_id": [diabetes.admission_source_id], 
        "time_in_hospital": [diabetes.time_in_hospital],
        "payer_code": [diabetes.payer_code],             
        "medical_specialty": [diabetes.medical_specialty],
        "num_lab_procedures": [diabetes.num_lab_procedures],
        "num_procedures": [diabetes.num_procedures],
        "num_medications": [diabetes.num_medications],
        "number_outpatient": [diabetes.number_outpatient],
        "number_emergency": [diabetes.number_emergency],
        "number_inpatient": [diabetes.number_inpatient],
        "diag_1": [diabetes.diag_1],                 
        "diag_2": [diabetes.diag_2],                 
        "diag_3": [diabetes.diag_3],
        "number_diagnoses": [diabetes.number_diagnoses],
        "max_glu_serum": [diabetes.max_glu_serum],          
        "A1Cresult": [diabetes.A1Cresult],              
        "metformin": [diabetes.metformin],              
        "repaglinide": [diabetes.repaglinide],            
        "nateglinide": [diabetes.nateglinide],            
        "chlorpropamide": [diabetes.chlorpropamide],         
        "glimepiride": [diabetes.glimepiride],            
        "acetohexamide": [diabetes.acetohexamide],          
        "glipizide": [diabetes.glipizide],              
        "glyburide": [diabetes.glyburide],              
        "tolbutamide": [diabetes.tolbutamide],            
        "pioglitazone": [diabetes.pioglitazone],          
        "rosiglitazone": [diabetes.rosiglitazone],          
        "acarbose": [diabetes.acarbose],               
        "miglitol": [diabetes.miglitol],               
        "troglitazone": [diabetes.troglitazone],           
        "tolazamide": [diabetes.tolazamide],             
        "examide": [diabetes.examide],                
        "citoglipton": [diabetes.citoglipton],            
        "insulin": [diabetes.insulin],                
        "glyburide-metformin": [diabetes.glyburide_metformin],     
        "glipizide-metformin": [diabetes.glipizide_metformin],    
        "glimepiride-pioglitazone": [diabetes.glimepiride_pioglitazone],
        "metformin-rosiglitazone": [diabetes.metformin_rosiglitazone],
        "metformin-pioglitazone": [diabetes.metformin_pioglitazone], 
        "change": [diabetes.change],                 
        "diabetesMed": [diabetes.diabetesMed]                  
    }

    df_to_predict = pd.DataFrame(data)
    print("Antes",df_to_predict)
    df_to_predict = prepare_data(df_to_predict)
    print("Despues",df_to_predict)

    # Hacer predicciones
    prediction = model.predict(df_to_predict)
    res = prediction[0].item()
    strres = None
    if res == 2:
        strres = '>30'
    elif res == 1:
        strres = '<30'
    else:
        strres = 'NO'

    return{"Readmitted": strres}
    

def prepare_data(df): 

    # Drop features
    df = df.drop(['weight', 'payer_code', 'medical_specialty', 'examide', 'citoglipton'], axis=1)

    # calculate change times through 23 kinds of medicines
    # high change times refer to higher prob to readmit
    # 'num_med_changed' to counts medicine change
    medicine = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'glipizide', 'glyburide',
                'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'insulin', 'glyburide-metformin', 'tolazamide',
                'metformin-pioglitazone', 'metformin-rosiglitazone', 'glimepiride-pioglitazone', 'glipizide-metformin',
                'troglitazone', 'tolbutamide', 'acetohexamide']

    for med in medicine:
        tmp = med + 'temp'
        df[tmp] = df[med].apply(lambda x: 1 if (x == 'Down' or x == 'Up') else 0)

    # two new feature
    df['num_med_changed'] = 0
    for med in medicine:
        tmp = med + 'temp'
        df['num_med_changed'] += df[tmp]
        del df[tmp]

    for i in medicine:
        df[i] = df[i].replace('Steady', 1)
        df[i] = df[i].replace('No', 0)
        df[i] = df[i].replace('Up', 1)
        df[i] = df[i].replace('Down', 1)
    df['num_med_taken'] = 0
    for med in medicine:
        df['num_med_taken'] = df['num_med_taken'] + df[med]

    # encode race
    df['race'] = df['race'].replace('Asian', 0)
    df['race'] = df['race'].replace('AfricanAmerican', 1)
    df['race'] = df['race'].replace('Caucasian', 2)
    df['race'] = df['race'].replace('Hispanic', 3)
    df['race'] = df['race'].replace('Other', 4)
    df['race'] = df['race'].replace('?', 4)

    # map
    df['A1Cresult'] = df['A1Cresult'].replace('None', -99)  # -1 -> -99
    df['A1Cresult'] = df['A1Cresult'].replace('>8', 1)
    df['A1Cresult'] = df['A1Cresult'].replace('>7', 1)
    df['A1Cresult'] = df['A1Cresult'].replace('Norm', 0)

    df['max_glu_serum'] = df['max_glu_serum'].replace('>200', 1)
    df['max_glu_serum'] = df['max_glu_serum'].replace('>300', 1)
    df['max_glu_serum'] = df['max_glu_serum'].replace('Norm', 0)
    df['max_glu_serum'] = df['max_glu_serum'].replace('None', -99)  # -1 -> -99

    df['change'] = df['change'].replace('No', 0)
    df['change'] = df['change'].replace("Ch", 1)

    df['gender'] = df['gender'].replace('Male', 1)
    df['gender'] = df['gender'].replace('Female', 0)

    df['diabetesMed'] = df['diabetesMed'].replace('Yes', 1)
    df['diabetesMed'] = df['diabetesMed'].replace('No', 0)


    age_dict = {'[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35, '[40-50)': 45, '[50-60)': 55, '[60-70)': 65,
                '[70-80)': 75, '[80-90)': 85, '[90-100)': 95}
    df['age'] = df.age.map(age_dict)
    df['age'] = df['age'].astype('int64')


    # simplify
    # admission_type_id : [2, 7] -> 1, [6, 8] -> 5
    a, b = [2, 7], [6, 8]
    for i in a:
        df['admission_type_id'] = df['admission_type_id'].replace(i, 1)
    for j in b:
        df['admission_type_id'] = df['admission_type_id'].replace(j, 5)

    # discharge_disposition_id : [6, 8, 9, 13] -> 1, [3, 4, 5, 14, 22, 23, 24] -> 2,
    #                            [12, 15, 16, 17] -> 10, [19, 20, 21] -> 11, [25, 26] -> 18
    a, b, c, d, e = [6, 8, 9, 13], [3, 4, 5, 14, 22, 23, 24], [12, 15, 16, 17], \
                    [19, 20, 21], [25, 26]
    for i in a:
        df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(i, 1)
    for j in b:
        df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(j, 2)
    for k in c:
        df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(k, 10)
    # data of died patients have been dropped
    # for p in d:
    #     df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(p, 11)
    for q in e:
        df['discharge_disposition_id'] = df['discharge_disposition_id'].replace(q, 18)

    # admission_source_id : [3, 2] -> 1, [5, 6, 10, 22, 25] -> 4,
    #                       [15, 17, 20, 21] -> 9, [13, 14] -> 11
    a, b, c, d = [3, 2], [5, 6, 10, 22, 25], [15, 17, 20, 21], [13, 14]
    for i in a:
        df['admission_source_id'] = df['admission_source_id'].replace(i, 1)
    for j in b:
        df['admission_source_id'] = df['admission_source_id'].replace(j, 4)
    for k in c:
        df['admission_source_id'] = df['admission_source_id'].replace(k, 9)
    for p in d:
        df['admission_source_id'] = df['admission_source_id'].replace(p, 11)


    #  Classify Diagnoses by ICD-9
    df.loc[df['diag_1'].str.contains('V', na=False), ['diag_1']] = 0
    df.loc[df['diag_1'].str.contains('E', na=False), ['diag_1']] = 0

    df['diag_1'] = df['diag_1'].replace('?', -1)

    df['diag_1'] = pd.to_numeric(df['diag_1'], errors='coerce')

    for index, row in df.iterrows():
        if (row['diag_1'] >= 1 and row['diag_1'] <= 139):
            df.loc[index, 'diag_1'] = 1
        elif (row['diag_1'] >= 140 and row['diag_1'] <= 239):
            df.loc[index, 'diag_1'] = 2
        elif (row['diag_1'] >= 240 and row['diag_1'] <= 279):
            df.loc[index, 'diag_1'] = 3
        elif (row['diag_1'] >= 280 and row['diag_1'] <= 289):
            df.loc[index, 'diag_1'] = 4
        elif (row['diag_1'] >= 290 and row['diag_1'] <= 319):
            df.loc[index, 'diag_1'] = 5
        elif (row['diag_1'] >= 320 and row['diag_1'] <= 389):
            df.loc[index, 'diag_1'] = 6
        elif (row['diag_1'] >= 390 and row['diag_1'] <= 459):
            df.loc[index, 'diag_1'] = 7
        elif (row['diag_1'] >= 460 and row['diag_1'] <= 519):
            df.loc[index, 'diag_1'] = 8
        elif (row['diag_1'] >= 520 and row['diag_1'] <= 579):
            df.loc[index, 'diag_1'] = 9
        elif (row['diag_1'] >= 580 and row['diag_1'] <= 629):
            df.loc[index, 'diag_1'] = 10
        elif (row['diag_1'] >= 630 and row['diag_1'] <= 679):
            df.loc[index, 'diag_1'] = 11
        elif (row['diag_1'] >= 680 and row['diag_1'] <= 709):
            df.loc[index, 'diag_1'] = 12
        elif (row['diag_1'] >= 710 and row['diag_1'] <= 739):
            df.loc[index, 'diag_1'] = 13
        elif (row['diag_1'] >= 740 and row['diag_1'] <= 759):
            df.loc[index, 'diag_1'] = 14
        elif (row['diag_1'] >= 760 and row['diag_1'] <= 779):
            df.loc[index, 'diag_1'] = 15
        elif (row['diag_1'] >= 780 and row['diag_1'] <= 799):
            df.loc[index, 'diag_1'] = 16
        elif (row['diag_1'] >= 800 and row['diag_1'] <= 999):
            df.loc[index, 'diag_1'] = 17

    df.loc[df['diag_2'].str.contains('V', na=False), ['diag_2']] = 0
    df.loc[df['diag_2'].str.contains('E', na=False), ['diag_2']] = 0

    df['diag_2'] = df['diag_2'].replace('?', -1)

    df['diag_2'] = pd.to_numeric(df['diag_2'], errors='coerce')

    for index, row in df.iterrows():
        if (row['diag_2'] >= 1 and row['diag_2'] <= 139):
            df.loc[index, 'diag_2'] = 1
        elif (row['diag_2'] >= 140 and row['diag_2'] <= 239):
            df.loc[index, 'diag_2'] = 2
        elif (row['diag_2'] >= 240 and row['diag_2'] <= 279):
            df.loc[index, 'diag_2'] = 3
        elif (row['diag_2'] >= 280 and row['diag_2'] <= 289):
            df.loc[index, 'diag_2'] = 4
        elif (row['diag_2'] >= 290 and row['diag_2'] <= 319):
            df.loc[index, 'diag_2'] = 5
        elif (row['diag_2'] >= 320 and row['diag_2'] <= 389):
            df.loc[index, 'diag_2'] = 6
        elif (row['diag_2'] >= 390 and row['diag_2'] <= 459):
            df.loc[index, 'diag_2'] = 7
        elif (row['diag_2'] >= 460 and row['diag_2'] <= 519):
            df.loc[index, 'diag_2'] = 8
        elif (row['diag_2'] >= 520 and row['diag_2'] <= 579):
            df.loc[index, 'diag_2'] = 9
        elif (row['diag_2'] >= 580 and row['diag_2'] <= 629):
            df.loc[index, 'diag_2'] = 10
        elif (row['diag_2'] >= 630 and row['diag_2'] <= 679):
            df.loc[index, 'diag_2'] = 11
        elif (row['diag_2'] >= 680 and row['diag_2'] <= 709):
            df.loc[index, 'diag_2'] = 12
        elif (row['diag_2'] >= 710 and row['diag_2'] <= 739):
            df.loc[index, 'diag_2'] = 13
        elif (row['diag_2'] >= 740 and row['diag_2'] <= 759):
            df.loc[index, 'diag_2'] = 14
        elif (row['diag_2'] >= 760 and row['diag_2'] <= 779):
            df.loc[index, 'diag_2'] = 15
        elif (row['diag_2'] >= 780 and row['diag_2'] <= 799):
            df.loc[index, 'diag_2'] = 16
        elif (row['diag_2'] >= 800 and row['diag_2'] <= 999):
            df.loc[index, 'diag_2'] = 17


    df.loc[df['diag_3'].str.contains('V', na=False), ['diag_3']] = 0
    df.loc[df['diag_3'].str.contains('E', na=False), ['diag_3']] = 0

    df['diag_3'] = df['diag_3'].replace('?', -1)

    df['diag_3'] = pd.to_numeric(df['diag_3'], errors='coerce')

    for index, row in df.iterrows():
        if (row['diag_3'] >= 1 and row['diag_3'] <= 139):
            df.loc[index, 'diag_3'] = 1
        elif (row['diag_3'] >= 140 and row['diag_3'] <= 239):
            df.loc[index, 'diag_3'] = 2
        elif (row['diag_3'] >= 240 and row['diag_3'] <= 279):
            df.loc[index, 'diag_3'] = 3
        elif (row['diag_3'] >= 280 and row['diag_3'] <= 289):
            df.loc[index, 'diag_3'] = 4
        elif (row['diag_3'] >= 290 and row['diag_3'] <= 319):
            df.loc[index, 'diag_3'] = 5
        elif (row['diag_3'] >= 320 and row['diag_3'] <= 389):
            df.loc[index, 'diag_3'] = 6
        elif (row['diag_3'] >= 390 and row['diag_3'] <= 459):
            df.loc[index, 'diag_3'] = 7
        elif (row['diag_3'] >= 460 and row['diag_3'] <= 519):
            df.loc[index, 'diag_3'] = 8
        elif (row['diag_3'] >= 520 and row['diag_3'] <= 579):
            df.loc[index, 'diag_3'] = 9
        elif (row['diag_3'] >= 580 and row['diag_3'] <= 629):
            df.loc[index, 'diag_3'] = 10
        elif (row['diag_3'] >= 630 and row['diag_3'] <= 679):
            df.loc[index, 'diag_3'] = 11
        elif (row['diag_3'] >= 680 and row['diag_3'] <= 709):
            df.loc[index, 'diag_3'] = 12
        elif (row['diag_3'] >= 710 and row['diag_3'] <= 739):
            df.loc[index, 'diag_3'] = 13
        elif (row['diag_3'] >= 740 and row['diag_3'] <= 759):
            df.loc[index, 'diag_3'] = 14
        elif (row['diag_3'] >= 760 and row['diag_3'] <= 779):
            df.loc[index, 'diag_3'] = 15
        elif (row['diag_3'] >= 780 and row['diag_3'] <= 799):
            df.loc[index, 'diag_3'] = 16
        elif (row['diag_3'] >= 800 and row['diag_3'] <= 999):
            df.loc[index, 'diag_3'] = 17


    numerics = ['race', 'age', 'time_in_hospital', 'num_medications', 'number_diagnoses',
                'num_med_changed', 'num_med_taken', 'number_inpatient', 'number_outpatient', 'number_emergency',
                'num_procedures', 'num_lab_procedures']
    
    feature_set = ['race', 'gender', 'age',
                   'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
                   'time_in_hospital', 'num_lab_procedures',
                   'num_procedures',
                   'num_medications', 'number_outpatient', 'number_emergency',
                   'number_inpatient', 'diag_1', 'number_diagnoses',
                   'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide',
                   'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
                   'tolbutamide',
                   'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
                   'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin',
                   'glipizide-metformin', 'glimepiride-pioglitazone',
                   'metformin-rosiglitazone', 'metformin-pioglitazone', 'change',
                   'diabetesMed', 'num_med_changed', 'num_med_taken']
    
    X_pred = df[feature_set]    

    return X_pred
