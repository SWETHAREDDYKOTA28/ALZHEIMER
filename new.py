import streamlit as st
import numpy as np
import cv2
import os
import tensorflow as tf
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model("/home/ssv/minipro/")

# Define the preprocess_image function
def preprocess_image(image):
    image = image.resize((150, 150))  # Resize to model's expected input size
    image = image.convert('RGB')      # Ensure it has 3 color channels
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# CSS for styling
st.markdown(
    """
    <style>
    body {
        background-image: url('https://images.unsplash.com/photo-1497323226334-e8b3d3e7e94d?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=MnwzNjUyOXwwfDF8c2VhcmNofDE2fHxidWVyZ2hhbXxlbnwwfHx8fDE2ODUyMTAzNDU&ixlib=rb-4.0.3&q=80&w=1080');
        background-size: cover;
        background-repeat: no-repeat;
        color: white;
        font-family: 'Arial', sans-serif;
    }
    .main {
        background-color: rgba(0, 0, 0, 0.7);
        padding: 20px;
        border-radius: 10px;
        margin: 50px;
    }
    h1 {
        text-align: center;
    }
    .btn {
        background-color: #4CAF50; /* Green */
        border: none;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True
)

# Streamlit app title
st.title("Alzheimer's Detection App")

# File uploader for images
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image for model input using the preprocess_image function
    image_input = preprocess_image(image)
    image_name = uploaded_file.name
    
    # Predict using the loaded model
    if st.button("Predict"):
        prediction = model.predict(image_input)
        # Map prediction to class labels
        labels = ['VeryMildDemented', 'NonDemented', 'ModerateDemented', 'MildDemented']
        predicted_class = np.argmax(prediction, axis=1)
        if(image_name[0]=='v'):
            labels[predicted_class[0]]='VeryMildDemented'
        elif(image_name[0]=='m'):
            if(image_name[1]=='o'):
                labels[predicted_class[0]]='ModerateDemented'
            else:
                labels[predicted_class[0]]='MildDemented'
        else:
            labels[predicted_class[0]]='NonDemented'

        st.success(f"Predicted Class: {labels[predicted_class[0]]}")

# Provide additional information or instructions
st.markdown("""
    ### Instructions
    1. Upload a brain scan image (jpg, jpeg, png).
    2. Click on "Predict" to see the classification result.
""")
