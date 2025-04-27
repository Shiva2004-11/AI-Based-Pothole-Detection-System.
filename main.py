import os
import cv2
import torch
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import datetime

# Load pre-trained models
cnn_model = load_model("pothole_classifier.h5")  # Load trained CNN model
yolo_model = YOLO("yolov8n.pt")  # Load YOLOv8 model

# Initialize history storage
if "detection_history" not in st.session_state:
    st.session_state.detection_history = []

# Image processing function
def process_image(image):
    image_resized = cv2.resize(image, (224, 224)) / 255.0  # Normalize
    image_resized = np.expand_dims(image_resized, axis=0)
    return image_resized

# Pothole detection function (without boundary analysis)
def detect_potholes(image):
    results = yolo_model(image)
    return image, results

# Generate detailed report
def generate_pdf():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Pothole Detection Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    for idx, entry in enumerate(st.session_state.detection_history):
        pdf.cell(0, 10, f"Detection {idx + 1}", ln=True)
        pdf.cell(0, 10, f"Date: {entry['timestamp']}", ln=True)
        pdf.cell(0, 10, f"Classification: {entry['label']}", ln=True)
        pdf.ln(5)
    
    pdf_file = "pothole_report.pdf"
    pdf.output(pdf_file)
    return pdf_file

# LiDAR-style visualization
def lidar_3d_visualization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    x, y = np.meshgrid(np.linspace(0, width, width), np.linspace(0, height, height))
    z = gray

    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Viridis')])
    fig.update_layout(title='LiDAR-style 3D Visualization of Pothole Image', autosize=False,
                      width=700, height=500,
                      margin=dict(l=65, r=50, b=65, t=90))
    st.plotly_chart(fig)

# Streamlit UI
st.sidebar.title("Navigation")
pages = ["Home", "Pothole Detection", "Detection History", "Statistics & Insights", "Report Generation", "Complaint Report", "Feedback & Improvement", "About & Team"]
page = st.sidebar.radio("Go to", pages)

if page == "Home":
    st.title("Pothole Detection System")
    st.write("This system detects potholes using deep learning models. Upload an image or use the webcam to get started.")
    st.image("C:/sem 6/DL-CAT-2/pothole new/pothole.gif", caption="How pothole detection works", use_column_width=True)

elif page == "Pothole Detection":
    option = st.radio("Choose input method:", ("Upload Image", "Use Webcam"))
    
    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            detected_image, results = detect_potholes(image)
            processed_img = process_image(image)
            prediction = cnn_model.predict(processed_img)
            label = "Pothole Detected" if prediction > 0.5 else "No Pothole"
            st.image(image, channels="BGR")
            st.write(f"**Classification Result:** {label}")
            st.session_state.detection_history.append({"timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "label": label})
            lidar_3d_visualization(image)

    elif option == "Use Webcam":
        img_file_buffer = st.camera_input("Capture Image")
        if img_file_buffer is not None:
            image = Image.open(img_file_buffer)
            image = np.array(image)
            detected_image, results = detect_potholes(image)
            processed_img = process_image(image)
            prediction = cnn_model.predict(processed_img)
            label = "Pothole Detected" if prediction > 0.5 else "No Pothole"
            st.image(image, channels="BGR")
            st.write(f"**Classification Result:** {label}")
            st.session_state.detection_history.append({"timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "label": label})
            lidar_3d_visualization(image)  # Added LiDAR visualization here

elif page == "Detection History":
    st.title("Detection History")
    if st.session_state.detection_history:
        df = pd.DataFrame(st.session_state.detection_history)
        st.table(df)
    else:
        st.write("No detection history available.")

elif page == "Statistics & Insights":
    st.title("Statistics & Insights")
    if st.session_state.detection_history:
        df = pd.DataFrame(st.session_state.detection_history)
        total_images = len(df)
        pothole_detected = df[df['label'] == "Pothole Detected"].shape[0]
        pothole_percentage = (pothole_detected / total_images) * 100 if total_images else 0
        st.write(f"Total images analyzed: {total_images}")
        st.write(f"Potholes detected: {pothole_detected} ({pothole_percentage:.2f}%)")
        fig = px.pie(df, names='label', title='Pothole Detection Distribution')
        st.plotly_chart(fig)
    else:
        st.write("No data available.")

elif page == "Report Generation":
    st.title("Generate Report")
    if st.session_state.detection_history:
        pdf_file = generate_pdf()
        with open(pdf_file, "rb") as file:
            st.download_button("Download Report", file, file_name=pdf_file)
    else:
        st.write("No data available to generate a report.")

elif page == "Complaint Report":
    st.title("File a Complaint")
    st.write("If you wish to file a complaint regarding potholes, click the button below.")
    if st.button("File Complaint"):
        st.markdown("[Click here to file a complaint](https://www.tnrsa.tn.gov.in/tnscrb/?utm_source=chatgpt.com)", unsafe_allow_html=True)

elif page == "Feedback & Improvement":
    st.title("Feedback & Improvement")
    feedback = st.text_area("Provide feedback about false positives/negatives or suggestions for improvement:")
    if st.button("Submit Feedback"):
        st.write("Thank you for your feedback!")

elif page == "About & Team":
    st.title("About & Team")
    st.write("This project was developed to detect potholes using deep learning techniques.")
    st.write("Team Members: Shiva Palaksha SG, Sibiyenthal K, Remote Ranjith LK")
