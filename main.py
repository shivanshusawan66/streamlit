import os
import pandas as pd
import streamlit as st
import joblib

# Custom CSS
css = '''
<style>
    .block-container {
        padding-top: 3rem;
    }

    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:26px;
    font-weight:500;
    }
    .stTabs [data-baseweb="tab-list"] {
        display: flex;  /* Use flexbox to arrange tabs */
        justify-content: space-between; 
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)

# Load the pre-trained model
@st.cache_resource
def load_model(model_path):
    """Helper function to load the pre-trained model."""
    model = joblib.load(model_path)
    return model

# Load dataset function
def load_dataset(file_path):
    """Helper function to load the dataset."""
    try:
        data = pd.read_csv("dataset/diabetes.csv")
        return data
    except FileNotFoundError:
        st.error(f"Dataset not found at {'dataset/diabetes.csv'}")
        return None

# Function to download the dataset
def download_data(file_path):
    """Download dataset as a CSV file."""
    with open(file_path, "rb") as f:
        st.download_button(
            label="Download Dataset", 
            data=f,
            file_name=os.path.basename(file_path),
            mime="text/csv",
            help="Click here to download the full dataset"
        )

# Function to reset the input fields
def reset_inputs():
    # Reset all input fields to their default values
    st.session_state.pregnancies = 0.0
    st.session_state.glucose = 0.0
    st.session_state.bloodPressure = 0.0
    st.session_state.skinThickness = 0.0
    st.session_state.insulin = 0.0
    st.session_state.bmi = 0.0
    st.session_state.diabetesPedigreeFunction = 0.0
    st.session_state.age = 0

# Initialize session state variables if they don't exist
if 'pregnancies' not in st.session_state:
    st.session_state.pregnancies = 0.0
if 'glucose' not in st.session_state:
    st.session_state.glucose = 0.0
if 'bloodPressure' not in st.session_state:
    st.session_state.bloodPressure = 0.0
if 'skinThickness' not in st.session_state:
    st.session_state.skinThickness = 0.0
if 'insulin' not in st.session_state:
    st.session_state.insulin = 0.0
if 'bmi' not in st.session_state:
    st.session_state.bmi = 0.0
if 'diabetesPedigreeFunction' not in st.session_state:
    st.session_state.diabetesPedigreeFunction = 0.0
if 'age' not in st.session_state:
    st.session_state.age = 0

# Tabs for navigation
tabs = st.tabs(["HOME", "ABOUT ME", "DIAGNOSIS"])

# Home Page
def home():
    with tabs[0]:
        st.header("Diabetes Prediction App")
        st.subheader("Welcome to the Diabetes Prediction Tool")
        
        st.write("""
        This application uses machine learning algorithms to predict the likelihood of diabetes in individuals based on certain health parameters.
        
        #### Features:
        - **User-friendly Interface**: Easily input your health parameters for quick predictions.
        - **Instant Results**: Get immediate feedback on your diabetes risk.
        - **Downloadable Dataset**: Access the complete dataset for your reference.
        
        #### How it Works:
        1. Enter your health data in the provided fields.
        2. Click on the **Predict** button to see your results.
        3. Optionally, reset the form using the **Reset** button.
        
        #### Get Started:
        Navigate to the **Diagnosis** section to start using the tool.
        """)
        
        st.image("img/symptoms.jpg", use_column_width=True)  # Replace with an appropriate image path

# About Us Page
def about():
    with tabs[1]:
        st.subheader("About Me")
        st.write("##### Hello! I'm Shivanshu Sawan")
        st.write("I am a final-year engineering student at UIET, Panjab University, specializing in Computer Science and Engineering.")
        
        st.write("I am passionate about leveraging technology to solve real-world problems. My interests include machine learning, software development, and data science.")
        
        st.write("Feel free to connect with me on my social media:")
        
        # GitHub link
        st.markdown("[GitHub](https://github.com/shivanshusawan66)")  # Replace with your GitHub link
        
        # LinkedIn link
        st.markdown("[LinkedIn](https://www.linkedin.com/in/shivanshu-sawan/)")  # Replace with your LinkedIn link

# Diagnosis Page
def diagnosis_page():
    with tabs[2]:
        st.subheader("Diabetes Prediction App")

        # File input for dataset
        dataset_path = "dataset/diabetes.csv"  # Change to your dataset path

        # Load and display dataset
        data = load_dataset(dataset_path)
        if data is not None:
            st.write("Here are the first 5 entries of the dataset:")
            st.dataframe(data.head())  # Display the first 5 rows of the dataset
            
            # Provide a download button for the dataset
            download_data(dataset_path)
        
        # Load Model
        model_path = "models/logistic_regression_model.pkl"  # Use forward slashes for paths
        model = load_model(model_path)

        # User inputs for prediction
        st.subheader("Enter patient details:")
        
        # Reset button 
        if st.button('Reset Data', key='reset_button'):
            reset_inputs()  # Call the reset function
        
        # Create input fields linked to session state
        pregnancies = st.number_input('Pregnancies', value=st.session_state.pregnancies, key='pregnancies')
        glucose = st.number_input('Glucose', value=st.session_state.glucose, key='glucose')
        bloodPressure = st.number_input('Blood Pressure', value=st.session_state.bloodPressure, key='bloodPressure')
        skinThickness = st.number_input('Skin Thickness', value=st.session_state.skinThickness, key='skinThickness')
        insulin = st.number_input('Insulin', value=st.session_state.insulin, key='insulin')
        bmi = st.number_input('BMI', value=st.session_state.bmi, key='bmi')
        diabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', value=st.session_state.diabetesPedigreeFunction, key='diabetesPedigreeFunction')
        age = st.number_input('Age', value=st.session_state.age, key='age')

        # Predict Button
        if st.button('Predict', key='predict_button'):
            with st.spinner('Processing...'):
                pred = model.predict([[pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, diabetesPedigreeFunction, age]])
                result = "Positive" if pred == [1] else "Negative"

                # Display result using st.success or st.error
                if result == "Positive":
                    st.error(f"The prediction is: **{result}**")
                else:
                    st.success(f"The prediction is: **{result}**")

# Page Routing
home()
about()
diagnosis_page()
