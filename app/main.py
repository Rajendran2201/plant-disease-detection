import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st


working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction.h5"
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))
disease_names = json.load(open(f"{working_dir}/disease_name.json"))
disease_remedies = json.load(open(f"{working_dir}/remedy.json"))
#disease_descriptions = json.load(open(f"{working_dir}/descriptions.json"))


# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]  # converting int into str
    return predicted_class_index,predicted_class_name

# Function to find the name of the disease
def predict_disease_name(predicted_class_index, disease_names):
    disease_name = ""
    for key, value in disease_names.items():
        if int(key) == predicted_class_index:
            disease_name = value
            break  
    return disease_name

# Function to find the description of the disease
def predict_disease_description(predicted_class_index, disease_descriptions):
    pass
    
# Function to find the remedy for the disease
def predict_disease_remedy(predicted_class_index, disease_remedies):
    disease_remedy = ""
    for key, value in disease_remedies.items():
        if int(key) == predicted_class_index:
            disease_remedy = value
            break  
    return disease_remedy


healthy_plants =[3, 4, 6, 10, 14, 17, 19, 22, 23, 24, 27, 37]

st.title('🍃 Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col2:
        resized_img = image.resize((150, 150))
        st.image(resized_img)
        button = st.button('Detect the disease')

    with col1:
        if button:
            # Preprocess the uploaded image and predict the class
            predicted_class_index, predicted_class_name = predict_image_class(model, uploaded_image, class_indices)
            # st.success(f'Prediction: {str(predicted_class_name)}')
            disease_name = predict_disease_name(predicted_class_index, disease_names)
            # printing the disease name 
            if predicted_class_index not in healthy_plants:
              st.markdown(f'<h1 style="color: red;">{disease_name}</h1>', unsafe_allow_html=True)
            else:
                st.markdown(f'<h1 style="color: green;">{disease_name}</h1>', unsafe_allow_html=True)


#disease_description = predict_disease_description(predicted_class_index, disease_descriptions)


# printing the description of the disease
#st.subheader("Description:")
#st.write(disease_description)
# printing the remedy of the disease
if 'predicted_class_index' in locals():
    disease_remedy = predict_disease_remedy(predicted_class_index, disease_remedies)
    st.subheader("Remedy:")
    st.write(disease_remedy)

            