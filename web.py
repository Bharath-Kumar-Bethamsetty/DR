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

menu = ["Home", "User Guide", "Upload Image and Predict", "About", 'Contact us' ]
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
    st.subheader("Welcome to the Diabetic Retinopathy Classification app.")
    y = '''Diabetic retinopathy is a diabetes complication that affects the eyes. It is caused by damage to the blood vessels in the retina, which can lead to vision impairment or even blindness if left untreated.

To tackle this problem, various research and development projects have been undertaken to detect and diagnose diabetic retinopathy at an early stage. These projects utilize a range of techniques, including computer vision and machine learning, to analyze retinal images and identify signs of the disease.

One approach involves developing deep learning models to analyze retinal images and detect signs of diabetic retinopathy. These models are trained on large datasets of retinal images labeled with diagnostic information, allowing them to learn to identify early signs of the disease with high accuracy.

Another approach involves using telemedicine to screen patients for diabetic retinopathy. This involves capturing retinal images using specialized cameras and transmitting them to ophthalmologists or other medical professionals for analysis. This approach can improve access to screening and diagnosis for patients who may not have access to specialized medical facilities.

Overall, these projects aim to improve early detection and diagnosis of diabetic retinopathy, which can lead to earlier treatment and better outcomes for patients. '''
	
    st.write(y)
    st.write("-----------")
    st.subheader("Identify signs of diabetic retinopathy in eye images")
    st.write("Diabetic retinopathy is the leading cause of blindness in the working-age population of the developed world. It is estimated to affect over 93 million people.")
    st.image("retina.jpg")
    x = '''
	The US Center for Disease Control and Prevention estimates that 29.1 million people in the US have diabetes and the World Health Organization estimates that 347 million people have the disease worldwide. Diabetic Retinopathy (DR) is an eye disease associated with long-standing diabetes. Around 40% to 45% of Americans with diabetes have some stage of the disease. Progression to vision impairment can be slowed or averted if DR is detected in time, however this can be difficult as the disease often shows few symptoms until it is too late to provide effective treatment.

Currently, detecting DR is a time-consuming and manual process that requires a trained clinician to examine and evaluate digital color fundus photographs of the retina. By the time human readers submit their reviews, often a day or two later, the delayed results lead to lost follow up, miscommunication, and delayed treatment.'''
    st.write(x)
	


	
	
	
	
    
   
if choice == "Upload Image and Predict":
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
            
        
       


if choice == "About":
    st.subheader("About")
    
    s = """
    
    The aim of a diabetic retinopathy classification project is to develop a computer vision system that can accurately classify diabetic retinopathy in retinal images. Diabetic retinopathy is a complication of diabetes that can lead to vision loss and blindness if not detected and treated early.

The project aims to use machine learning algorithms to analyze retinal images and identify signs of diabetic retinopathy such as microaneurysms, hemorrhages, and exudates. By accurately classifying the level of diabetic retinopathy in an image, the system can help clinicians prioritize patients for treatment and monitor disease progression over time.

The ultimate goal of a diabetic retinopathy classification project is to improve the quality of care for patients with diabetes by providing a more efficient and accurate method of screening for diabetic retinopathy, which can ultimately prevent vision loss and blindness.
    
    """
    
    st.info(s)
if choice == 'Contact us':
    st. header("Under guidance of Mr. K. E. Naresh Kumar  M.Tech., (Ph.D.) ") 
    st. success (" RGMCET") 
    st. subheader("Contact Details") 
    st. warning("B.Bharath Kumar") 
    st. success("bk337810@gmail.com") 
    st. warning ("P.G.Jaswanth Reddy") 
    st. success("jaswanthre561mb@gmail.com") 
    
    
    
if choice == "User Guide":
	st.header("WELCOME TO USER-GUIDE")
	st.subheader("Goto upload image session and just drap or drop your fundus image into it and wait for the prediction")
	st.subheader("*------------------------------------------------------------*")
	st.subheader("--> if you get the result as below")
	st.success("congratulations you are safe!! 😁👍")
	st.subheader("It means that your are safe and you are all perfect with your health")
	st.subheader("*------------------------------------------------------------*")
	st.subheader("--> Or if you get the result as below")
	st.warning("⚠️ Proper medication is required!!!")
	st.subheader("It means that you need to take care of your health and activities, because the model predicted that, in future you might suffer from diabetic retinopathy. And suggested that, its better to consult a doctor/specialist.")
	st.subheader("*------------------------------------------------------------*")
    
