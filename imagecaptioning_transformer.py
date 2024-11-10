import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import json
import os
import random
import re
from io import BytesIO

# Load your model and tokenizer here
# (you can adapt the model loading code from your original script)
from tensorflow.keras.preprocessing.text import Tokenizer

# Constants for model (these should be defined based on your previous setup)
MAX_LENGTH = 40
VOCABULARY_SIZE = 10000
EMBEDDING_DIM = 512
UNITS = 512

# Define the tokenizer and the model (ensure this is based on your code)
# Assuming the model and tokenizer are already loaded as 'caption_model'

# Define helper functions
def load_image_from_path(img_path):
    img = tf.io.read_file(img_path)  # Read the image file
    img = tf.io.decode_jpeg(img, channels=3)  # Decode as RGB (3 channels)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)  # Convert to float32
    img = tf.image.resize(img, (299, 299))  # Resize image (adjust to model input size)
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return img

def generate_caption(img_path):
    img = load_image_from_path(img_path)
    img = tf.expand_dims(img, axis=0)
    img_embed = caption_model.cnn_model(img)
    img_encoded = caption_model.encoder(img_embed, training=False)

    y_inp = '[start]'
    for i in range(MAX_LENGTH - 1):
        tokenized = tokenizer([y_inp])[:, :-1]
        mask = tf.cast(tokenized != 0, tf.int32)
        pred = caption_model.decoder(
            tokenized, img_encoded, training=False, mask=mask
        )
        pred_idx = np.argmax(pred[0, i, :])
        pred_word = idx2word(pred_idx).numpy().decode('utf-8')
        if pred_word == '[end]':
            break
        y_inp += ' ' + pred_word

    y_inp = y_inp.replace('[start] ', '')
    return y_inp

# Streamlit UI setup
st.title('Image Caption Generator with Transformer')
st.write("Upload an image to generate a caption.")

# Image upload section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Save the uploaded image temporarily
    temp_image_path = '/tmp/uploaded_image.jpg'
    img.save(temp_image_path)

    # Generate caption using the model
    caption = generate_caption(temp_image_path)

    # Display the generated caption
    st.write(f"Predicted Caption: {caption}")

# If you want to add more functionality like downloading the generated caption, you can do it below:
st.sidebar.write("#### Instructions:")
st.sidebar.write("1. Upload an image from your local machine.")
st.sidebar.write("2. Wait for the model to generate a caption based on the image.")
st.sidebar.write("3. The caption will be displayed below the image.")

# Optionally, you can also save the model's weights or download links
# st.download_button(label="Download Model Weights", data=open("image_captioning_transformer_weights.h5", "rb").read(), file_name="image_captioning_transformer_weights.h5")
