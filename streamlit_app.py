import streamlit as st

st.title("🎈Thitkhar App")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
import streamlit as st
from PIL import Image
import io
import cv2
import numpy as np
import tensorflow as tf  # Assuming TensorFlow for the models

st.title("Image Input with Face Detection and Recognition")

# Create a radio button for image input options
option = st.radio("Choose an option:", ("Browse Image", "Capture Image"))

# Load pre-trained face recognition models
def load_model_1():
    model_path = "Own_Dataset_GoogleNet_16_50.h5"  # Replace with your actual model file name
    return tf.keras.models.load_model(model_path)

def load_model_2():
    model_path = "Own_Dataset_FRM_16_50.h5"  # Replace with your actual model file name
    return tf.keras.models.load_model(model_path)

# Load the models
model_1 = load_model_1()
model_2 = load_model_2()

# Function to perform face detection
def detect_faces(image):
    # Convert PIL Image to OpenCV format (numpy array)
    image_np = np.array(image)

    # Convert RGB to BGR (OpenCV uses BGR format)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Load a pre-trained face detection model (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Convert image to grayscale for face detection
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Store detected face regions
    face_regions = []
    for (x, y, w, h) in faces:
        face_regions.append(image_np[y:y+h, x:x+w])
        cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert back to RGB for Streamlit display
    image_with_faces = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    return image_with_faces, face_regions, len(faces)

# Function to predict face identity using two models
def predict_face_identity(face_region, model_1, model_2):
    # Preprocess face_region for the models
    face = cv2.resize(face_region, (128, 128))  # Resize to match model input size
    face = np.expand_dims(face, axis=0)  # Add batch dimension
    face = face / 255.0  # Normalize pixel values

    # Get predictions from both models
    pred_1 = np.argmax(model_1.predict(face), axis=1)
    pred_2 = np.argmax(model_2.predict(face), axis=1)

    # Get label from class
    if pred_1[0] == "Class 0":
        pred_1_label = "Kaung Zaw Hein"
    
    if pred_1[0] == "Class 1":
        predict_1_label = "Khant Nay Linn Tun"

    if pred_1[0] == "Class 2":
        pred_1_label = "Min Thiha Kyaw"

    if pred_1[0] == "Class 3":
        pred_1_label = "Nay Phone Htoo"
    
    if pred_1[0] == "Class 4":
        pred_1_label = "Pyae Phyo Han"

    if pred_1[0] == "Class 5":
        pred_1_label = "Win Htet Oo"

    if pred_2[0] == "Class 0":
        pred_2_label = "Kaung Zaw Hein"
    
    if pred_2[0] == "Class 1":
        pred_2_label = "Khant Nay Linn Tun"

    if pred_2[0] == "Class 2":
        pred_2_label = "Min Thiha Kyaw"

    if pred_2[0] == "Class 3":
        pred_2_label = "Nay Phone Htoo"
    
    if pred_2[0] == "Class 4":
        pred_2_label = "Pyae Phyo Han"

    if pred_2[0] == "Class 5":
        pred_2_label = "Win Htet Oo"

    return f"Model 1 (GoogleNet) predicts: Class {pred_1_label}", f"Model 2 (VGG_16) predicts: Class {pred_2_label}"

if option == "Browse Image":
    # Upload image from local system
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Convert uploaded file to PIL Image
        image = Image.open(uploaded_file)

        # Perform face detection
        image_with_faces, face_regions, face_count = detect_faces(image)

        # Display the uploaded and processed images
        st.image(image, caption="Uploaded Image")
        st.image(image_with_faces, caption="Image with Detected Faces")

        # Show the number of detected faces
        st.write(f"Detected {face_count} face(s) in the image.")

        # Predict face identities
        for i, face in enumerate(face_regions):
            pred_1, pred_2 = predict_face_identity(face, model_1, model_2)
            st.write(f"Face {i+1}:")
            st.write(pred_1)
            st.write(pred_2)

elif option == "Capture Image":
    # Capture image using the camera
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        # Convert image buffer to bytes and then to PIL Image
        bytes_data = img_file_buffer.getvalue()
        image = Image.open(io.BytesIO(bytes_data))

        # Perform face detection
        image_with_faces, face_regions, face_count = detect_faces(image)

        # Display the captured and processed images
        st.image(image, caption="Captured Image")
        st.image(image_with_faces, caption="Image with Detected Faces")

        # Show the number of detected faces
        st.write(f"Detected {face_count} face(s) in the image.")

        # Predict face identities
        for i, face in enumerate(face_regions):
            pred_1, pred_2 = predict_face_identity(face, model_1, model_2)
            st.write(f"Face {i+1}:")
            st.write(pred_1)
            st.write(pred_2)
