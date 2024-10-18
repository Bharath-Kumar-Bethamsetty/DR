import streamlit as st
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from time import sleep

# Set page configuration
st.set_page_config(page_title="Diabetic Retinopathy Classification", page_icon=":eyes:", layout='wide')

# Title and Sidebar
st.title("Diabetic Retinopathy Classification")
st.sidebar.title("Menu")

menu = ["Home", "User Guide", "Upload Image and Get Prediction", "About", "Contact Us", "Map"]
choice = st.sidebar.selectbox("Select an option", menu)

# CSS for sidebar customization
st.markdown(
    """
    <style>
        .sidebar .sidebar-content {
            width: 375px;
            font-size: 1rem;  /* Adjust font size for better readability */
        }
        .big-font {
            font-size: 20px;  /* Use a larger font size for emphasis */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Home Page
if choice == "Home":
    st.subheader("Welcome to the Diabetic Retinopathy Classification app.")
    st.write("""
        Diabetic retinopathy is a diabetes complication that affects the eyes. It is caused by damage to the blood vessels in the retina,
        which can lead to vision impairment or even blindness if left untreated.

        Various research and development projects aim to detect and diagnose diabetic retinopathy at an early stage using computer vision
        and machine learning techniques to analyze retinal images.
    """)
    st.image("retina.jpg", use_column_width=True)

    st.write("""
        The CDC estimates that 29.1 million people in the US have diabetes, and 40% to 45% of Americans with diabetes have some stage of 
        diabetic retinopathy. Early detection is crucial as the disease often shows few symptoms until it is too late to provide effective treatment.
    """)
    st.image("retina2.jpg", use_column_width=True)

    st.write("""
        All images are categorized according to the severity/stage of diabetic retinopathy using the train.csv file provided. 
        You will find five directories with the respective images:
        - 0 - No DR
        - 1 - Mild
        - 2 - Moderate
        - 3 - Severe
        - 4 - Proliferate DR
    """)

    st.subheader("Understanding the Stages of Diabetic Retinopathy")
    st.write("""
        Elevated blood sugar and pressure can damage the delicate blood vessels of the retina, leading to diabetic retinopathy. 
        In the early stages, vision loss may be prevented, but as the condition advances, it becomes harder to prevent vision loss.
    """)
    st.image("retina3.jpg", use_column_width=True)

    st.subheader("Types of DR")
    st.image("types.jpg", use_column_width=True)
    st.subheader("Sample Images")
    st.image("samples.jpg", use_column_width=True)

# Upload Image and Prediction Page
if choice == "Upload Image and Get Prediction":
    model = load_model('cnn.h5')
    class_labels = ['NO-DR', 'DR']  # Adjusted for clarity

    def predict(image):
        img = Image.open(image).convert('RGB')
        img = img.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        predictions = model.predict(img)
        label = np.argmax(predictions[0])
        return class_labels[label]

    def main():
        uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.subheader("Analyzing the uploaded image...")
            with st.spinner('Processing, please wait...'):
                time.sleep(3)
                
            st.subheader("Getting results...")

            progress = st.progress(0)
            for i in range(100):
                progress.progress(i + 1)
                sleep(0.05)

            label = predict(uploaded_file)
            if label == 'DR':
                st.warning("⚠️ The uploaded image is classified as: **Diabetic Retinopathy**")
                st.subheader("**Proper medication is required!**")
            else:
                st.success("🎉 The uploaded image is classified as: **No Diabetic Retinopathy**")
                st.subheader("**Congratulations, you are safe!**")
                st.balloons()

    if __name__ == '__main__':
        main()

# About Page
if choice == "About":
    st.subheader("About")
    st.write("""
        The aim of the diabetic retinopathy classification project is to develop a computer vision system that can accurately classify 
        diabetic retinopathy in retinal images. By using machine learning algorithms, we can analyze retinal images and identify signs of 
        diabetic retinopathy such as microaneurysms, hemorrhages, and exudates. This can help clinicians prioritize patients for treatment 
        and monitor disease progression over time, ultimately improving patient care and preventing vision loss.
    """)

# Contact Us Page
if choice == "Contact Us":
    st.header("Under the guidance of Mr. K. E. Naresh Kumar, M.Tech., (Ph.D.)")
    st.success("RGMCET")
    st.subheader("Contact Details")
    contacts = {
        "B. Bharath Kumar": "bk337810@gmail.com",
        "P. G. Jaswanth Reddy": "jaswanthre561mb@gmail.com",
        "P. Karthik": "karthiklucky9988@gmail.com",
        "M. Vijay Babu": "vijaymanda046@gmail.com",
    }
    for name, email in contacts.items():
        st.warning(name)
        st.success(email)

# User Guide Page
if choice == "User Guide":
    st.header("Welcome to User Guide")
    st.subheader("Upload your fundus image by dragging and dropping it into the upload section and wait for the prediction.")
    st.subheader("If you receive the message below:")
    st.success("🎉 Congratulations, you are safe!")
    st.subheader("It means that you are in good health.")
    st.subheader("If you receive this message:")
    st.warning("⚠️ Proper medication is required!")
    st.subheader("You need to take care of your health as the model predicts potential risks of diabetic retinopathy. It is advisable to consult a specialist.")

# Map Page
if choice == "Map":
    st.map()
  
