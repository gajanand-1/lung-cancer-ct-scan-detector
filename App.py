import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_efficientnet

# --- App Configuration ---
st.set_page_config(
    page_title="Ensemble Image Classifier",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- Dummy File Creation (for first-time run) ---
# This part creates placeholder models if your actual models aren't found.
# It ensures the app is runnable out-of-the-box for demonstration.
# IMPORTANT: Replace these dummy files with your actual trained models.
def create_dummy_models_if_needed():
    """Checks for model files and creates dummy ones if they don't exist."""
    if not os.path.exists('vgg16_model.keras'):
        st.warning("VGG16 model not found. Creating a dummy model as a placeholder.")
        dummy_vgg16 = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(4, activation='softmax')
        ])
        dummy_vgg16.save('vgg16_model.keras')
        st.toast("Dummy VGG16 model created.")

    if not os.path.exists('efficientnet_model.keras'):
        st.warning("EfficientNet model not found. Creating a dummy model as a placeholder.")
        dummy_efficientnet = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(4, activation='softmax')
        ])
        dummy_efficientnet.save('efficientnet_model.keras')
        st.toast("Dummy EfficientNet model created.")

# --- Model Loading ---
@st.cache_resource
def load_models():
    """Loads the VGG16 and EfficientNet models from disk."""
    try:
        vgg16_model = tf.keras.models.load_model('vgg16_model.keras')
        efficientnet_model = tf.keras.models.load_model('efficientnet_model.keras')
        return vgg16_model, efficientnet_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.error("Please ensure your 'vgg16_model.keras' and 'efficientnet_model.keras' files are in the same directory as app.py.")
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

# Create dummy models on first run if needed
create_dummy_models_if_needed()

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

