import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Function to load user data from JSON file
def load_user_data():
    if os.path.exists('user_data.json'):
        with open('user_data.json', 'r') as f:
            return json.load(f)
    return {}

# Function to save user data to JSON file
def save_user_data(user_data):
    with open('user_data.json', 'w') as f:
        json.dump(user_data, f)

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.radio("Select Page", ["Home", "Disease Recognition", "Login", "Register", "About"])

# Add a logout button in the sidebar
if st.session_state.get('logged_in', False):
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.success("You have logged out successfully.")
        app_mode = "Home"  # Redirect to home page after logout

# Create an empty container
placeholder = st.empty()

# Initialize session state to track login status
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Load user data from the file
user_data = load_user_data()

# Home page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# Disease Recognition page (only accessible if logged in)
elif app_mode == "Disease Recognition":
    if not st.session_state.logged_in:
        st.warning("Please log in to access Disease Recognition.")
        app_mode = "Login"  # Redirect to login page
    else:
        st.header("Disease Recognition")
        test_image = st.file_uploader("Choose an Image:")
        if st.button("Show Image"):
            st.image(test_image, width=4, use_container_width=True)
        
        # Predict button
        if st.button("Predict"):
            st.image(test_image, width=4, use_container_width=True)
            st.snow()
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]
            st.success(f"Model is Predicting it's a {class_name[result_index]}")

# Registration form
elif app_mode == "Register":
    with placeholder.form("register"):
        st.markdown("#### Register a new account")
        first_name = st.text_input("First Name")
        last_name = st.text_input("Last Name")
        mobile_number = st.text_input("Mobile Number", max_chars=11)
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Register")

    # Validate registration
    if submit:
        if len(mobile_number) != 11:
            st.error("Please enter a valid 11-digit mobile number.")
        elif mobile_number in user_data:
            st.error("This mobile number is already registered.")
        elif password != confirm_password:
            st.error("Passwords do not match.")
        else:
            # Store the user data (in a real app, this should be stored securely in a database)
            user_data[mobile_number] = {"first_name": first_name, "last_name": last_name, "password": password}
            save_user_data(user_data)  # Save the updated user data to the file
            st.success("Registration successful! You can now log in.")

# Login form
elif app_mode == "Login":
    with placeholder.form("login"):
        st.markdown("#### Enter your credentials")
        login_mobile_number = st.text_input("Mobile Number", max_chars=11)
        login_password = st.text_input("Password", type="password")
        login_submit = st.form_submit_button("Login")

    if login_submit:
        if login_mobile_number in user_data and user_data[login_mobile_number]["password"] == login_password:
            st.session_state.logged_in = True
            st.success("Login successful! You can now access the Disease Recognition page.")
            app_mode = "Disease Recognition"  # Redirect to Disease Recognition page
        else:
            st.error("Login failed: Invalid mobile number or password.")

# About page (just as an example for the other pages)
elif app_mode == "About":
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
                This dataset consists of about 87K RGB images of healthy and diseased crop leaves, categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation set, preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purposes.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)
                """)
