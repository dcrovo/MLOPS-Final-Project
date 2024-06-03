import streamlit as st
import requests

# URL of the FastAPI backend
API_URL = "http://10.43.101.155/predict/"

st.title("Diabetes Prediction API")
st.write("MLOps - Proyecto 3")
st.write("Autores:")
st.write("    Daniel Crovo (dcrovo@javeriana.edu.co)")
st.write("    Carlos Trujillo (ca.trujillo@javeriana.edu.co)")

with st.form("prediction_form"):
    # Default values set as per your curl example
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        race = st.selectbox("Race", ["Caucasian", "Asian", "AfricanAmerican", "Hispanic", "Other"], index=0)
        age = st.selectbox("Age", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"], index=0)
        num_lab_procedures = st.number_input("Number of Lab Procedures", value=41, min_value=0, max_value=200)
        number_outpatient = st.number_input("Number of Outpatient Visits", value=0, min_value=0, max_value=100)
    
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"], index=0)
        weight = st.selectbox("Weight", ["[0-25)", "[25-50)", "[50-75)", "[75-100)", "[100-125)", "[125-150)", "[150-175)", "[175-200)"], index=0)
        num_procedures = st.number_input("Number of Other Procedures", value=0, min_value=0, max_value=100)
        number_emergency = st.number_input("Number of Emergency Visits", value=0, min_value=0, max_value=100)
    
    with col3:
        admission_type_id = st.number_input("Admission Type ID", value=6, min_value=1, max_value=8)
        time_in_hospital = st.number_input("Time in Hospital", value=1, min_value=1, max_value=50)
        num_medications = st.number_input("Number of Medications", value=1, min_value=0, max_value=100)
        number_inpatient = st.number_input("Number of Inpatient Visits", value=0, min_value=0, max_value=100)
    
    with col4:
        discharge_disposition_id = st.number_input("Discharge Disposition ID", value=25, min_value=1, max_value=30)
        admission_source_id = st.number_input("Admission Source ID", value=7, min_value=1, max_value=25)
        diag_1 = st.text_input("Primary Diagnosis Code", value="250.83")
        diag_2 = st.text_input("Secondary Diagnosis Code", value="250.01")
    
    diag_3 = st.text_input("Additional Diagnosis Code", value="255")
    number_diagnoses = st.number_input("Number of Diagnoses", value=1, min_value=1, max_value=100)
    max_glu_serum = st.selectbox("Max Glucose Serum", ["None", ">200", ">300", "Norm"], index=1)
    A1Cresult = st.selectbox("A1C Result", ["None", ">7", ">8", "Norm"], index=2)
    change = st.selectbox("Change of Medications", ["No", "Ch"], index=0)
    diabetesMed = st.selectbox("On Diabetes Medication", ["No", "Yes"], index=0)

    submit_button = st.form_submit_button("Predict")
    
    if submit_button:
        data = {
            "race": race, "gender": gender, "age": age, "weight": weight,
            "admission_type_id": admission_type_id, "discharge_disposition_id": discharge_disposition_id,
            "admission_source_id": admission_source_id, "time_in_hospital": time_in_hospital,
            "payer_code": "BC", "medical_specialty": "Cardiology",
            "num_lab_procedures": num_lab_procedures, "num_procedures": num_procedures,
            "num_medications": num_medications, "number_outpatient": number_outpatient,
            "number_emergency": number_emergency, "number_inpatient": number_inpatient,
            "diag_1": diag_1, "diag_2": diag_2, "diag_3": diag_3,
            "number_diagnoses": number_diagnoses, "max_glu_serum": max_glu_serum,
            "A1Cresult": A1Cresult, "metformin": "No", "repaglinide": "No",
            "nateglinide": "No", "chlorpropamide": "No", "glimepiride": "No",
            "acetohexamide": "No", "glipizide": "No", "glyburide": "No",
            "tolbutamide": "No", "pioglitazone": "No", "rosiglitazone": "No",
            "acarbose": "No", "miglitol": "No", "troglitazone": "No",
            "tolazamide": "No", "examide": "No", "citoglipton": "No",
            "insulin": "No", "glyburide_metformin": "No", "glipizide_metformin": "No",
            "glimepiride_pioglitazone": "No", "metformin_rosiglitazone": "No",
            "metformin_pioglitazone": "No", "change": "No", "diabetesMed": "No",
            "readmitted": "NO"
        }
        
        try:
            response = requests.post(API_URL, json=data)
            response.raise_for_status()  # This will raise for HTTP errors.
            prediction = response.json()
            st.success(f"Prediction: {prediction['Readmitted']}")
        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP error occurred: {e.response.status_code} {e.response.reason}")
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {str(e)}")
        except ValueError:
            st.error("Failed to decode JSON from response.")
