import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model once to avoid reloading on every prediction
@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model("BrainTumor10Epoch.h5")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Function for TensorFlow model prediction
def model_prediction(model, test_image):
    image = Image.open(test_image)
    image = image.resize((64, 64))
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    return prediction

# Function to display medical recommendations
def display_recommendations(prediction_index):
    if prediction_index == 0:
        st.info("No Tumor Detected. However, regular MRI checkups are recommended for confirmation.")
    else:
        st.warning("Tumor Detected! Please consult a neurologist or oncologist immediately for further analysis.")

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #f8f8f8;
        }
        .stButton>button {
            color: white;
            background-color: #0073e6;
            border: none;
            padding: 10px 16px;
            font-size: 16px;
            border-radius: 8px;
        }
        h1, h2, h3 {
            color: #333333;
        }
        .result-frame {
            background-color: #ffffff;
            border: 2px solid #e6e6e6;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Load the TensorFlow model
model = load_model()

# Main Header
st.markdown("<h1 style='text-align: center; color: darkred;'>Brain Tumor Detection</h1>", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.header("Welcome to the Brain Tumor Classification System 🧠")
    st.image("c:\\lab1_dev\\salma_venv\\lab1\\dataset\\images.brain.jpg")
    st.markdown("""
    ### How It Works:
    1. **Upload Image:** Go to the **Disease Recognition** page and upload a brain MRI image.
    2. **Analysis:** Our system processes the image using Deep learning models to detect brain tumors.
    3. **Results:** View the results and receive recommendations for further medical action.
    
    Together, we can aid in early diagnosis and treatment!
    """)

# About Page
elif app_mode == "About":
    st.header("About Brain Tumors")
    st.markdown("""
    **Brain Tumors** are abnormal growths of cells in the brain. Tumors can be:
    - **Benign:** Non-cancerous.
    - **Malignant:** Cancerous and aggressive.

    #### Common Symptoms:
    - Persistent headaches
    - Nausea and vomiting
    - Seizures
    - Cognitive and memory issues
    - Vision problems
    - Weakness or numbness

    #### Prevention Tips:
    - Avoid exposure to harmful chemicals
    - Maintain a healthy lifestyle
    - Monitor family medical history
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("Brain Tumor Recognition 📊")
    test_image = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])
    
    if test_image is not None:
        # Display the uploaded image
        st.image(test_image, use_container_width=True)

        if st.button("Predict"):
            st.write("Processing... Please wait.")
            prediction = model_prediction(model, test_image)
            class_names = ['No Tumor', 'Tumor Detected']

            # Extract prediction results
            predicted_index = np.argmax(prediction)
            probability = prediction[0][predicted_index] * 100

            # Display results in a framed container
            st.markdown("""
            <div class="result-frame">
                <h2 style="text-align: center;">Prediction Results</h2>
                <p><b>Model Prediction:</b> {}</p>
                <p><b>Confidence:</b> {:.2f}%</p>
                <h3>Class Probabilities:</h3>
                <ul>
                    <li><b>No Tumor:</b> {:.2f}%</li>
                    <li><b>Tumor Detected:</b> {:.2f}%</li>
                </ul>
            </div>
            """.format(class_names[predicted_index], probability, prediction[0][0] * 100, prediction[0][1] * 100), unsafe_allow_html=True)

            # Medical recommendations
            display_recommendations(predicted_index)
st.write("Medical Recommendations:")