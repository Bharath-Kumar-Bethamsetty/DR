import streamlit as st
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from time import sleep




st.set_page_config(page_title="Diabetic Retinopathy Classification", page_icon=":eyes:")
st.title("Diabetic Retinopathy Classification")
st.sidebar.title("Menu")

menu = ["Home", "Upload Image","view dataset", "About"]
choice = st.sidebar.selectbox("Select an option", menu)



st.markdown(
    f'''
        <style>
            .sidebar .sidebar-content {{
                width: 375px;
                font-size: 100px;
            }}
        </style>
    ''',
    unsafe_allow_html=True
)



if choice == "Home":
    st.write("Welcome to the Diabetic Retinopathy Classification app.")
    
    
   
if choice == "Upload Image":
    model = load_model('cnn.h5')
    class_labels = ['DR', 'NO-DR']

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
            st.subheader("Please wait uploaded file is analysing")
            with st.spinner('processing please wait...'):
                time.sleep(5)
                
                
              
 


        
            st.subheader("getting results....")

            progress = st.progress(0)

            for i in range(0,101):
                progress.progress(i)
                sleep(0.05)
            



            

            
                
            st.image(image, caption='Uploaded Image',use_column_width=True)
            label = predict(uploaded_file)
            if label == 'DR':
                st.warning("The uploaded image is classified as : DR ")
                st.subheader("⚠️ Proper medication is required!!!")
            else:
                st.success("The uploaded image is classified as : NO-DR ")
                st.subheader("congratulations you are safe!! 😁👍")
                st.balloons()

    if __name__ == '__main__':
        main()
            
        
        
      
if choice == "view dataset":

    st.write("---Catogories of DR---")
    st.write("0 --> No DR stage")
    st.write("1 --> Mild stage")
    st.write("2 --> Moderate DR stage")
    st.write("3 --> Severe DR stage")
    st.write("4 --> Proliferative (greatly effected)")

    data = pd.read_csv("train.csv")
    st.write(data)


if choice == "About":
    st.subheader("About")
    
    s = """
    
    The aim of a diabetic retinopathy classification project is to develop a computer vision system that can accurately classify diabetic retinopathy in retinal images. Diabetic retinopathy is a complication of diabetes that can lead to vision loss and blindness if not detected and treated early.

The project aims to use machine learning algorithms to analyze retinal images and identify signs of diabetic retinopathy such as microaneurysms, hemorrhages, and exudates. By accurately classifying the level of diabetic retinopathy in an image, the system can help clinicians prioritize patients for treatment and monitor disease progression over time.

The ultimate goal of a diabetic retinopathy classification project is to improve the quality of care for patients with diabetes by providing a more efficient and accurate method of screening for diabetic retinopathy, which can ultimately prevent vision loss and blindness.
    
    """
    
    st.info(s)
