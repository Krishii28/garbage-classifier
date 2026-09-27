import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

st.set_page_config(page_title="Garbage Classifier", page_icon="♻️")
st.title("♻️ Garbage Classifier - 94% Accuracy")
st.write("Upload garbage image to classify")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("garbage_model.keras")
    with open("classes.json") as f:
        classes = json.load(f)
    return model, classes

model, classes = load_model()

file = st.file_uploader("Image upload karo", type=["jpg","png","jpeg"])
if file:
    img = Image.open(file).convert("RGB").resize((224,224))
    st.image(img)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(img, dtype=np.float32)[None,...])
    pred = model.predict(arr)[0]
    idx = int(np.argmax(pred))
    st.success(f"Result: {classes[idx]} ({pred[idx]*100:.1f}%)")
