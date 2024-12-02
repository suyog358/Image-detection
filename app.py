import os
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Load the trained model
model = load_model("model.h5")

# Ensure the 'uploads' directory exists and is a directory
upload_dir = "uploads"
if os.path.exists(upload_dir):
    if not os.path.isdir(upload_dir):
        os.remove(upload_dir)  # Remove the file if it's not a directory
        os.makedirs(upload_dir)  # Create the directory
else:
    os.makedirs(upload_dir)

# Prediction function
def predict_image(file_path):
    size = (150, 150)
    image = Image.open(file_path)
    image = ImageOps.grayscale(image)  # Convert to grayscale
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image).reshape(1, 150, 150, 1)
    image_array = image_array / 255.0  # Normalize
    prediction = model.predict(image_array)
    return "Cancer Detected" if prediction[0][0] > 0.5 else "No Cancer"

# Streamlit interface
st.title('Brain Tumor Prediction')
st.write("Upload an image to predict if a brain tumor is present or not.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded image to the 'uploads' directory
    img_path = os.path.join(upload_dir, uploaded_file.name)
    try:
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display uploaded image
        st.image(img_path, caption="Uploaded Image", use_column_width=True)

        # Make prediction
        result = predict_image(img_path)
        st.subheader(f"Prediction: {result}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
