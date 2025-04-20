import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from datetime import datetime

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Load user data
def load_user_data():
    if os.path.exists('user_data.json'):
        with open('user_data.json', 'r') as f:
            return json.load(f)
    return {}

# Save user data
def save_user_data(user_data):
    with open('user_data.json', 'w') as f:
        json.dump(user_data, f)

# Log user inputs
def log_image_input(mobile_number, image_path, prediction):
    if not os.path.exists('user_inputs.json'):
        with open('user_inputs.json', 'w') as f:
            json.dump([], f)
    with open('user_inputs.json', 'r') as f:
        inputs = json.load(f)
    input_data = {
        "mobile_number": mobile_number,
        "image_path": image_path,
        "prediction": prediction,
        "timestamp": str(datetime.now())
    }
    inputs.append(input_data)
    with open('user_inputs.json', 'w') as f:
        json.dump(inputs, f)

# Save uploaded image
def save_uploaded_file(uploaded_file):
    upload_dir = 'uploaded_images/'
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Admin credentials
ADMIN_CREDENTIALS = {"admin": "admin"}

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.radio("Select Page", ["Home", "Disease Recognition", "Login", "Register", "Admin Dashboard", "About"])

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False

# Logout buttons
if st.session_state.logged_in:
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.is_admin = False
        st.success("You have logged out successfully.")
        app_mode = "Home"

# Placeholder for forms
placeholder = st.empty()

# Load user data
user_data = load_user_data()

# Home page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    st.image("home_page.jpeg", use_container_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Upload an image of a plant to detect potential diseases using our ML model.

    Navigate to **Disease Recognition** to get started!
    """)

# Disease Recognition page
elif app_mode == "Disease Recognition":
    if not st.session_state.logged_in:
        st.warning("Please log in to access Disease Recognition.")
    else:
        st.header("Disease Recognition")
        test_image = st.file_uploader("Choose an Image:")
        if st.button("Predict") and test_image:
            image_path = save_uploaded_file(test_image)
            st.image(image_path, use_container_width=True)
            st.snow()
            result_index = model_prediction(image_path)
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                          'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                          'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                          'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                          'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                          'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                          'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                          'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy',
                          'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
                          'Strawberry___healthy','Sweet_Potato_Healthy','Tomato___Bacterial_spot', 'Sweet_Potato_Healthy',
                          'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
                          'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
                          'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
            prediction = class_name[result_index]
            st.success(f"Model predicts: {prediction}")
            log_image_input(st.session_state.user_mobile, image_path, prediction)

# Register page
elif app_mode == "Register":
    with placeholder.form("register"):
        st.markdown("#### Register a new account")
        first_name = st.text_input("First Name")
        last_name = st.text_input("Last Name")
        mobile_number = st.text_input("Mobile Number", max_chars=11)
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Register")
    if submit:
        if len(mobile_number) != 11:
            st.error("Enter a valid 11-digit mobile number.")
        elif mobile_number in user_data:
            st.error("Mobile number already registered.")
        elif password != confirm_password:
            st.error("Passwords do not match.")
        else:
            user_data[mobile_number] = {"first_name": first_name, "last_name": last_name, "password": password}
            save_user_data(user_data)
            st.success("Registration successful! Please login.")

# Login page
elif app_mode == "Login":
    with placeholder.form("login"):
        st.markdown("#### Enter your credentials")
        login_mobile_number = st.text_input("Mobile Number", max_chars=11)
        login_password = st.text_input("Password", type="password")
        login_submit = st.form_submit_button("Login")
    if login_submit:
        if login_mobile_number in user_data and user_data[login_mobile_number]["password"] == login_password:
            st.session_state.logged_in = True
            st.session_state.user_mobile = login_mobile_number
            st.success("Login successful!")
        else:
            st.error("Invalid mobile number or password.")

# Admin Dashboard
elif app_mode == "Admin Dashboard":
    if not st.session_state.is_admin:
        admin_username = st.text_input("Admin Username")
        admin_password = st.text_input("Admin Password", type="password")
        submit_admin = st.button("Login as Admin")
        if submit_admin:
            if ADMIN_CREDENTIALS.get(admin_username) == admin_password:
                st.session_state.is_admin = True
                st.success("Admin login successful!")
            else:
                st.error("Invalid admin credentials.")
    else:
        st.header("Admin Dashboard")
        if st.button("Admin Logout"):
            st.session_state.is_admin = False
            st.success("Admin logged out successfully.")
            app_mode = "Home"
        if st.button("View Model Output (User Predictions)"):
            st.subheader("User Predictions and Image Uploads")
            if os.path.exists('user_inputs.json'):
                with open('user_inputs.json', 'r') as f:
                    user_inputs = json.load(f)
                if user_inputs:
                    for entry in user_inputs:
                        st.write(f"User {entry['mobile_number']} uploaded image {entry['image_path']} at {entry['timestamp']}. Prediction: {entry['prediction']}")
                        if os.path.exists(entry['image_path']):
                            st.image(entry['image_path'], caption=f"Uploaded Image by {entry['mobile_number']}", use_container_width=True)
                        else:
                            st.warning(f"Image not found at path: {entry['image_path']}")
                else:
                    st.warning("No predictions available.")
            else:
                st.warning("No user input logs found.")
        st.subheader("User Information")
        for mobile, details in user_data.items():
            st.write(f"Mobile: {mobile}, Name: {details['first_name']} {details['last_name']}")

# About Page
elif app_mode == "About":
    st.header("About Us")
    st.markdown("""
    This Plant Disease Recognition System was built to help identify plant diseases from leaf images using AI.

    **Mission**: Assist farmers and agriculturists in protecting crops with fast, accurate diagnosis.

    **Team**: Developed by a group of dedicated software developers and AI enthusiasts passionate about agriculture.
    """)
