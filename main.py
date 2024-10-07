import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Working directory and model path
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Function to load and preprocess the image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to predict the class of an image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Streamlit App Layout
st.set_page_config(page_title="Plant Disease Classifier", layout="wide", initial_sidebar_state="collapsed")

# Updated CSS for the entire website styling and navbar
st.markdown("""
    <style>
        body {
            font-family: 'Helvetica Neue', sans-serif;
            background-color: #f0f2f6;
        }
        .main {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        }
        h1 {
            font-size: 3em;
            font-weight: 600;
            margin-bottom: 20px;
            color: #0071e3;
            text-align: center;
        }
        h3 {
            color: #333333;
            font-weight: 400;
            margin-bottom: 20px;
            text-align: center;
        }
        .stButton button {
            background-color: #0071e3;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 30px;
            border: none;
            transition: background-color 0.3s ease;
        }
        .stButton button:hover {
            background-color: #005bb5;
        }
        .stImage img {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 200px;
            margin-bottom: 20px;
        }
        .navbar {
            background-color: #0071e3;
            padding: 10px;
            color: white;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            display: flex;
            justify-content: space-around;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .navbar a {
            color: white;
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 4px;
        }
        .navbar a:hover {
            background-color: #005bb5;
        }
    </style>
    """, unsafe_allow_html=True)

# Navbar display at the top of the page (visual only, logic handled by query parameters)
st.markdown("""
    <div class="navbar">
        <a href="?page=Home">🏠 Home</a>
        <a href="?page=Plant%20Disease%20Detection">🌿 Plant Disease Detection</a>
        <a href="?page=About%20the%20Team">👨‍💻 About the Team</a>
    </div>
    """, unsafe_allow_html=True)

# Get query parameters to handle navigation
query_params = st.experimental_get_query_params()  # Change to `st.session_state.query_params` when updated

# Determine which page to load
page = query_params.get("page", ["Home"])[0]

# Home Page
if page == "Home":
    st.title('Welcome to the Plant Disease Classifier 🌿')
    st.subheader("Your AI solution for identifying plant diseases 🧠")
    st.markdown("""
    ### 🌟 Features:
    - Upload plant images and receive real-time disease predictions.
    - AI-powered model for accurate classification.
    - Easy-to-use interface with a clean and intuitive design.
    
    Navigate to the **Plant Disease Detection** page to upload and classify plant images, or visit the **About the Team** page to learn more about us! 😄
    """)

# Plant Disease Detection Page
elif page == "Plant Disease Detection":
    col1, col2 = st.columns([1, 5])

    with col1:
        st.image("Designer.png", width=100)  # Smaller logo next to title
    with col2:
        st.title('🌿 Plant Disease Detection')

    st.subheader("Effortlessly identify plant diseases with AI-driven accuracy 🤖")

    # Image uploader section
    uploaded_image = st.file_uploader("Upload an image to classify plant diseases", type=["jpg", "jpeg", "png"])

    # Processing and Prediction
    if uploaded_image is not None:
        image = Image.open(uploaded_image)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image.resize((250, 250)), caption="Uploaded Image", use_column_width=True)

        with col2:
            st.write("")
            st.write("")
            if st.button('🌟 Classify Image'):
                prediction = predict_image_class(model, uploaded_image, class_indices)
                st.markdown(f"<h3 style='text-align: center;'>🧬 Prediction: {str(prediction)}</h3>", unsafe_allow_html=True)

# About the Team Page
elif page == "About the Team":
    st.title('👨‍💻 About the Team')
    st.markdown("""
    ### Meet the Team 🚀
    We are a group of passionate studnet engineers working on cutting-edge AI solutions for agriculture. Our mission is to leverage AI to help farmers detect plant diseases early, leading to healthier crops and better yields.
    
    ### Team Members:
    - **Samridh Singh** - Industrial Internet of Things Engineer in Training🧠
    Our team believes in creating accessible, cost-effective tools for modern agriculture using AI and machine learning.
    """)

    # Team illustration placeholder (replace with actual images)
    st.image("Designer.png", width=400, caption="Our Team", use_column_width=True)