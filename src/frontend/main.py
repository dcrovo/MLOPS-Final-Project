import streamlit as st
import requests
import os
from glob import glob
from datetime import datetime

# URL of the FastAPI backend
API_URL = "http://10.43.101.155:8084"

st.title("Realtor Price Prediction API")
st.write("MLOps - Proyecto Final")
st.write("Autores:")
st.write("    Daniel Crovo (dcrovo@javeriana.edu.co)")
st.write("    Carlos Trujillo (ca.trujillo@javeriana.edu.co)")

# Fetch unique values for dropdowns
@st.cache_data(ttl=600)
def fetch_unique_values():
    response = requests.get(f"{API_URL}/unique_values/")
    response.raise_for_status()
    return response.json()

# Fetch unique values
unique_values = fetch_unique_values()


with st.form("prediction_form"):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        brokered_by = st.selectbox("Brokered By", unique_values['brokered_by'])
        status = st.selectbox("Status", ["sold", "for_sale"])
        bed = st.number_input("Number of Beds", min_value=1, max_value=10, value=2)
        bath = st.number_input("Number of Baths", min_value=1, max_value=10, value=1)
    
    with col2:
        acre_lot = st.number_input("Acre Lot", min_value=0.01, max_value=100.0, value=0.1)
        street = st.selectbox("Street", unique_values['street'])
        city = st.selectbox("City", unique_values['city'])
        state = st.selectbox("State", unique_values['state'])
    
    with col3:
        house_size = st.number_input("House Size (sq ft)", min_value=100, max_value=10000, value=960)
    
    submit_button = st.form_submit_button("Predict")
    
    if submit_button:
        data = {
            "brokered_by": brokered_by,
            "status": status,
            "bed": bed,
            "bath": bath,
            "acre_lot": acre_lot,
            "street": street,
            "city": city,
            "state": state,
            "house_size": house_size
        }
        
        try:
            response = requests.post(f"{API_URL}/predict/", json=data)
            response.raise_for_status()  # This will raise for HTTP errors.
            prediction = response.json()
            st.success(f"Prediction: {prediction['prediction']}")
        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP error occurred: {e.response.status_code} {e.response.reason}")
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {str(e)}")
        except ValueError:
            st.error("Failed to decode JSON from response.")
# Function to get the latest image file from a directory
def get_latest_image(directory):
    list_of_files = glob(os.path.join(directory, '*.png'))  # Adjust the extension as needed
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

# Directory where images are saved
image_directory = "./img/"

# Get the latest image file
latest_image = get_latest_image(image_directory)

# Display the latest image if it exists
if True:
    st.image('/opt/code/img/shap_random_forest_best_model_v5.png', caption='Latest SHAP Plot', use_column_width=True)
else:
    st.write("No images found.")