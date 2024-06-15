import streamlit as st
import pickle
import numpy as np

# Load the model from the pickle file
with open(r'C:\Users\Hemanth\OneDrive\Desktop\Ml_Deployment\Salary.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit app
def main():
    st.title("Model Prediction App")
    
    # Input fields
    st.write("Enter the input values:")
    input1 = st.number_input("Input 1", value=0.0)
    input2 = st.number_input("Input 2", value=0.0)
    input3 = st.number_input("Input 3", value=0.0)
    input4 = st.number_input("Input 4", value=0.0)
    input5 = st.number_input("Input 5", value=0.0)
    
    # Button to generate the output
    if st.button("Generate Output"):
        input_data = np.array([[input1, input2, input3, input4, input5]])
        prediction = model.predict(input_data)
        st.write("Prediction:", prediction[0])

if __name__ == '__main__':
    main()
