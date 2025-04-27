# AI-Based-Pothole-Detection-System.
🚧 Pothole Detection System:

Overview:
This project presents a Pothole Detection System that uses deep learning models for detecting potholes from uploaded images or live webcam captures.
It is built with Streamlit for a user-friendly web interface and integrates CNN and YOLOv8 models for classification and object detection.

Users can:

*Upload images or capture photos through the webcam
*Get pothole detection results with 3D visualization
*View detection history
*Analyze statistics and insights
*Generate downloadable PDF reports
*File complaints
*Provide feedback

Features:
✅ Upload an image for pothole detection
✅ Capture live images using the webcam
✅ CNN-based image classification
✅ YOLOv8-based pothole object detection
✅ LiDAR-style 3D visualization of road surfaces
✅ Store and view detection history
✅ Generate a detailed Pothole Detection Report in PDF format
✅ Analyze statistics and insights from detections
✅ File complaints via official portal
✅ Collect user feedback for system improvement

Tech Stack 🛠
Frontend: Streamlit
Deep Learning Models:
 *CNN (TensorFlow/Keras .h5 model)
 *YOLOv8 (Ultralytics)

Visualization:
 *OpenCV
 *Plotly
 *Matplotlib

Others:
 *FPDF for PDF generation
 *Pandas, NumPy for data handling
 *PIL for image processing

Dataset Description 📂
Uploaded Image Dataset:
 *Users upload road surface images.
 *Images undergo resizing (224x224) and normalization before being classified.
 *Predictions are made using both CNN (classification) and YOLOv8 (object detection).

Webcam Captured Dataset:
 *Images are captured in real-time using Streamlit's st.camera_input().
 *Captured images are converted into arrays, resized to 224x224, normalized, and processed live.
 *These dynamic images help in testing the robustness of the models against:
    *Different lighting conditions
    *Real-world road surface textures
    *Camera quality variations
 *No pre-labeling at capture time — used directly for live detection.

 Installation 🔧
Clone the repository:
git clone https://github.com/your-username/pothole-detection-system.git
cd pothole-detection-system.

Install the required packages:
pip install -r requirements.txt

Place the models:
 *Save your trained CNN model as pothole_classifier.h5.
 *Ensure yolov8n.pt (YOLOv8 Nano model) is present in the project folder.

Run the application:
streamlit run app.py

Folder Structure 📁:
pothole-detection-system/
│
├── pothole_classifier.h5     # Trained CNN model
├── yolov8n.pt                 # YOLOv8 object detection model
├── app.py                     # Main Streamlit app
├── requirements.txt           # Python package requirements
├── README.md                  # Project README
└── assets/                    # Images like pothole.gif, icons, etc. (Optional)

License 📜
This project is licensed under the MIT License.

Team ✨
*Shiva Palaksha SG
*Sibiyenthal K
*Ranjith LK



