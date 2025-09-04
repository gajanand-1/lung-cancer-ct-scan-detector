import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import requests
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_efficientnet

# --- App Configuration ---
st.set_page_config(
    page_title="Ensemble Image Classifier",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- Function to download a file from a URL ---
def download_file(url, filename):
    """Downloads a file from a URL to a local path if it doesn't exist."""
    if not os.path.exists(filename):
        try:
            with st.spinner(f"Downloading model: {os.path.basename(filename)}... This may take a few moments."):
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(filename, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
            st.toast(f"Downloaded {os.path.basename(filename)} successfully.")
        except Exception as e:
            st.error(f"Error downloading {filename}: {e}")
            st.error("Please check the URL and your internet connection. You may need to restart the app after fixing the issue.")
            st.stop()


# --- Model URLs and Paths (PASTE YOUR COPIED LINKS HERE) ---
VGG16_URL = "PASTE_THE_LINK_FOR_VGG16_YOU_COPIED_HERE"
EFFICIENTNET_URL = "PASTE_THE_LINK_FOR_EFFICIENTNET_YOU_COPIED_HERE"

VGG16_PATH = 'vgg16_model.keras'
EFFICIENTNET_PATH = 'efficientnet_model.keras'

# --- Download models if they don't exist ---
download_file(VGG16_URL, VGG16_PATH)
download_file(EFFICIENTNET_URL, EFFICIENTNET_PATH)


# --- Model Loading ---
@st.cache_resource
def load_models():
    """Loads the VGG16 and EfficientNet models from disk."""
    try:
        vgg16_model = tf.keras.models.load_model(VGG16_PATH)
        efficientnet_model = tf.keras.models.load_model(EFFICIENTNET_PATH)
        return vgg16_model, efficientnet_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# --- Image Preprocessing ---
def preprocess_image(img, model_type, target_size=(224, 224)):
    """
    Preprocesses a PIL image for a given model type.
    
    Args:
        img (PIL.Image): The input image.
        model_type (str): 'vgg16' or 'efficientnet'.
        target_size (tuple): The target image dimensions.

    Returns:
        A preprocessed image tensor.
    """
    # The pre-trained models expect 3 channels. If the image is grayscale (like a CT scan),
    # we convert it to RGB by duplicating the single channel three times.
    if img.mode != "RGB":
        img = img.convert("RGB")
        
    img = img.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)

    if model_type == 'vgg16':
        return preprocess_input_vgg16(img_array_expanded)
    elif model_type == 'efficientnet':
        return preprocess_input_efficientnet(img_array_expanded)
    else:
        raise ValueError("Unknown model type")

# --- Main App Interface ---
st.title("🧠 Ensemble Vision Classifier")
st.markdown("Upload an image and this app will use two models (VGG16 and EfficientNet) to predict its class.")

# Define the class names your model predicts
CLASS_NAMES = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']

# Load the models
vgg16_model, efficientnet_model = load_models()

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and vgg16_model is not None and efficientnet_model is not None:
    # Display the uploaded image
    pil_image = Image.open(uploaded_file)
    st.image(pil_image, caption="Your Uploaded Image", use_column_width=True)

    # Prediction button
    if st.button("Classify Image", type="primary"):
        with st.spinner("Analyzing the image..."):
            # Preprocess the image for both models, getting shape dynamically
            vgg16_input_shape = vgg16_model.input_shape[1:3]
            efficientnet_input_shape = efficientnet_model.input_shape[1:3]
            
            preprocessed_vgg16 = preprocess_image(pil_image, 'vgg16', target_size=vgg16_input_shape)
            preprocessed_efficientnet = preprocess_image(pil_image, 'efficientnet', target_size=efficientnet_input_shape)

            # Make predictions
            probs_vgg16 = vgg16_model.predict(preprocessed_vgg16)
            probs_efficientnet = efficientnet_model.predict(preprocessed_efficientnet)
            
            # Average the probabilities
            avg_probs = (probs_vgg16 + probs_efficientnet) / 2.0
            
            # Get final prediction
            predicted_class_index = np.argmax(avg_probs)
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence = avg_probs[0][predicted_class_index]

        # Display results
        st.success(f"**Prediction Complete!**")
        col1, col2 = st.columns(2)
        col1.metric("Predicted Class", predicted_class_name)
        col2.metric("Confidence", f"{confidence:.2%}")

        # Display detailed probabilities in an expander
        with st.expander("Show Detailed Probabilities"):
            prob_data = {
                "Class": CLASS_NAMES,
                "VGG16": [f"{p:.2%}" for p in probs_vgg16[0]],
                "EfficientNet": [f"{p:.2%}" for p in probs_efficientnet[0]],
                "**Average**": [f"**{p:.2%}**" for p in avg_probs[0]]
            }
            st.dataframe(prob_data, use_container_width=True)
            
st.markdown("---")
st.write("Built with ❤️ using Streamlit and TensorFlow.")

