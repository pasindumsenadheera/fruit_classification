import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('fruit_classification_mobilenetv2.h5')

# Define the class labels (update this with your dataset's class names)
class_names = ['apple', 'mango', 'banana', 'cherry', 'pineapple', 
               'strawberries', 'watermelon', 'avocado', 'kiwi', 'orange']

# Title and description
st.title("Fruit Classification App")
st.write("Upload an image of a fruit and let the model predict which fruit it is!")

# Upload the image
uploaded_file = st.file_uploader("Choose a fruit image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    # Preprocess the image to match the input shape of the model
    img = img.resize((100, 100))  # Resize to the input size used during training
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image

    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]

    # Display the result
    st.write(f"The model predicts this is a **{predicted_class}**!")
