import streamlit as st
import numpy as np
import pickle

# Title of the app
st.title("ML Model Integration")

# Sidebar for user input
st.sidebar.header("Input Parameters")
def user_input_features():
    param1 = st.sidebar.slider("Parameter 1", 0.0, 10.0, 5.0)
    param2 = st.sidebar.slider("Parameter 2", 0.0, 10.0, 5.0)
    param3 = st.sidebar.slider("Parameter 3", 0.0, 10.0, 5.0)
    data = {"param1": param1, "param2": param2, "param3": param3}
    return data

input_data = user_input_features()

# Display input data
st.subheader("Input Parameters")
st.write(input_data)

# Load your model (replace 'your_model.pkl' with your actual model file)
try:
    with open("your_model.pkl", "rb") as file:
        model = pickle.load(file)
    st.write("Model loaded successfully!")

    # Prepare data for prediction
    input_array = np.array([list(input_data.values())])
    prediction = model.predict(input_array)
    st.subheader("Prediction")
    st.write(prediction[0])

except FileNotFoundError:
    st.error("Model file not found! Please upload the 'your_model.pkl' file.")

# Footer
st.markdown("**Streamlit App for ML Model**")
