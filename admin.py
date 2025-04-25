import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from datetime import datetime

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
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
            class_name = [
                 {"রোগ": "আপেল স্ক্যাব", "কেন হয়": "আর্দ্র আবহাওয়া ও অতিরিক্ত পানির কারণে ছত্রাক জন্মায়।", "প্রতিকার": "আক্রান্ত পাতা ছেঁটে ফেলুন, কপার ছত্রাকনাশক স্প্রে করুন। নিয়মিত ছাঁটাই ও পানি নিষ্কাশনের ব্যবস্থা নিন।"},
                 {"রোগ": "আপেল ব্ল্যাক রট", "কেন হয়": "পচা ফল বা পুরনো ডালে ছত্রাক জন্ম নিয়ে ছড়ায়।","প্রতিকার": "পচা ফল ও ডাল কেটে ফেলুন, ছত্রাকনাশক ব্যবহার করুন। গাছের নিচে পড়ে থাকা ফল নিয়মিত পরিষ্কার করুন।"},
                 {"রোগ": "সিডার আপেল রস্ট","কেন হয়": "আপেল গাছের কাছে সিডার গাছ থাকলে ছত্রাক ছড়ায়।", "প্রতিকার": "সিডার গাছ সরান, ছত্রাকনাশক ব্যবহার করুন। কাছাকাছি সিডার গাছ না রাখলে রোগ হবে না।"}, 
                 'আপেল___সুস্থ',
                 'ব্লুবেরি___সুস্থ', 
                 {"রোগ": "চেরি পাউডারি মিলডিউ","কেন হয়": "শুষ্ক আবহাওয়ায় পাতায় ছত্রাকের স্তর জমে।", "প্রতিকার": "সালফার স্প্রে দিন, বাতাস চলাচলের ব্যবস্থা রাখুন। গাছ খুব ঘন না হলে রোগ কম হয়।"},
                 'চেরি (টকসহ)___সুস্থ', 
                 {"রোগ": "ভুট্টা সারকোসপোরা পাতা দাগ",  "কেন হয়": "গরম ও আর্দ্র আবহাওয়া।","প্রতিকার": "পাতা ছেঁটে ফেলুন এবং ছত্রাকনাশক স্প্রে করুন। সঠিক পানি সরবরাহ নিশ্চিত করুন।"},
                 {"রোগ": "ভুট্টা কমন রস্ট", "কেন হয়": "আর্দ্র ও ঠাণ্ডা আবহাওয়ায় ছত্রাকের বীজ ছড়ায়।",  "প্রতিকার": "রোগমুক্ত জাত লাগান, ছত্রাকনাশক ব্যবহার করুন। জমিতে আগের রোগমুক্ত ফসল চাষে সাহায্য পাবে।"}, 
                 {"রোগ": "ভুট্টা নর্দার্ন পাতা ব্লাইট", "কেন হয়": "আর্দ্র আবহাওয়া ও দীর্ঘ সময় বৃষ্টির কারণে ছত্রাক আক্রমণ।", "প্রতিকার": "গাছের ঘনত্ব কমিয়ে দিন, রোগমুক্ত জাত ব্যবহার করুন।"},
                 'ভুট্টা___সুস্থ',
                 {"রোগ": "আঙ্গুর ব্ল্যাক রট",  "কেন হয়": "আর্দ্র আবহাওয়া ও পচা ফল থেকে ছত্রাক ছড়ায়।","প্রতিকার": "আক্রান্ত ফল ও ডাল কেটে ফেলুন, ছত্রাকনাশক ব্যবহার করুন।"}, 
                 {"রোগ": "আঙ্গুর এসকা (ব্ল্যাক মিজলস)", "কেন হয়": "অতিরিক্ত আর্দ্রতা ও দীর্ঘকালীন উচ্চ তাপমাত্রা।","প্রতিকার": "আক্রান্ত লতা সরিয়ে ফেলুন, ছত্রাকনাশক স্প্রে করুন।"}, 
                 {"রোগ": "আঙ্গুর পাতা ব্লাইট (আইসারিওপসিস পাতা দাগ)",  "কেন হয়": "বৃষ্টি ও আর্দ্রতার কারণে ছত্রাক ছড়ায়।","প্রতিকার": "গাছের ডাল ছেঁটে ফেলুন, ছত্রাকনাশক স্প্রে করুন।"},
                 'আঙ্গুর___সুস্থ',
                 {"রোগ": "কমলা হুয়াংলংবিং (সাইট্রাস গ্রিনিং)","কেন হয়": "সাদা মাছি দ্বারা ভাইরাস সংক্রমণ।", "প্রতিকার": "সাদা মাছি নিয়ন্ত্রণ করুন, আক্রান্ত গাছ সরান।"}, 
                 {"রোগ": "পিচ ব্যাকটেরিয়াল স্পট", "কেন হয়": "আর্দ্র আবহাওয়া ও উচ্চ তাপমাত্রা।","প্রতিকার": "তামা ভিত্তিক ছত্রাকনাশক ব্যবহার করুন। আক্রান্ত গাছ সরান।"},
                 'পিচ___সুস্থ', 
                 {"রোগ": "বেল মরিচ ব্যাকটেরিয়াল স্পট", "কেন হয়": "শীতল ও আর্দ্র আবহাওয়া।","প্রতিকার": "বেল মরিচের ডাল ও পাতা ছেঁটে ফেলুন, কপার স্প্রে ব্যবহার করুন।"}, 
                 'বেল মরিচ___সুস্থ',
                 {"রোগ": "আলু আর্লি ব্লাইট",  "কেন হয়": "শীতল ও আর্দ্র আবহাওয়া।", "প্রতিকার": "অস্তিত্বশীল আলু জাত ব্যবহার করুন, সঠিক সার ও পানি ব্যবহার করুন।"}, 
                 {"রোগ": "আলু লেট ব্লাইট",  "কেন হয়": "ঠাণ্ডা ও আর্দ্র পরিবেশে ছত্রাকের বৃদ্ধি।", "প্রতিকার": "ছত্রাকনাশক স্প্রে করুন এবং জমিতে আগের রোগমুক্ত ফসল চাষ করুন।"},
                 'আলু___সুস্থ', 
                 'রাস্পবেরি___সুস্থ',
                 'সয়াবিন___সুস্থ', 
                 {"রোগ": "স্কোয়াশ পাউডারি মিলডিউ", "কেন হয়": "আর্দ্র আবহাওয়া ও কম বাতাস চলাচল।","প্রতিকার": "সালফার স্প্রে দিন, বাতাস চলাচলের ব্যবস্থা রাখুন।"}, 
                 {"রোগ": "স্ট্রবেরি পাতার স্কর্চ", "কেন হয়": "গরম ও আর্দ্র আবহাওয়া।", "প্রতিকার": "পাতা ছেঁটে ফেলুন এবং ছত্রাকনাশক স্প্রে করুন।"},
                 'স্ট্রবেরি___সুস্থ',
                 {"রোগ": "মিষ্টি আলু সুস্থ"}, 
                 {"রোগ": "টমেটো ব্যাকটেরিয়াল স্পট", "কেন হয়": "অতিরিক্ত আর্দ্রতা ও সঠিক পরিচর্যার অভাব।",  "প্রতিকার": "কপার ছত্রাকনাশক স্প্রে করুন।"},    
                 {"রোগ": "মিষ্টি আলু সুস্থ"},
                 {"রোগ": "টমেটো লেট ব্লাইট",  "কেন হয়": "গরম ও আর্দ্র আবহাওয়া।",  "প্রতিকার": "রোগমুক্ত জাত ব্যবহার করুন, ছত্রাকনাশক স্প্রে করুন।"},
                 {"রোগ": "টমেটো পাতার ছাঁচ", "কেন হয়": "ঠাণ্ডা ও আর্দ্র আবহাওয়া।",  "প্রতিকার": "আক্রান্ত অংশ ছেঁটে ফেলুন, ছত্রাকনাশক প্রয়োগ করুন।"}, 
                 {"রোগ": "টমেটো সেপ্টোরিয়া দাগ", "কেন হয়": "আর্দ্র আবহাওয়া।", "প্রতিকার":  "আক্রান্ত অংশ কেটে ফেলুন, ছত্রাকনাশক ব্যবহার করুন।"},
                 {"রোগ": "টমেটো স্পাইডার মাইট",  "কেন হয়": "গরম ও শুকনো আবহাওয়া।", "প্রতিকার": "নিম তেল স্প্রে করুন এবং পোকা নিয়ন্ত্রণ করুন।"}, 
                 {"রোগ": "টমেটো টার্গেট স্পট", "কেন হয়": "গরম ও আর্দ্র আবহাওয়া।",  "প্রতিকার": "ছত্রাকনাশক স্প্রে এবং রোগমুক্ত জাত ব্যবহার করুন।"},
                 {"রোগ": "টমেটো ইয়েলো লিফ কার্ল ভাইরাস", "কেন হয়": "সাদা মাছি দ্বারা ভাইরাস ছড়ায়।","প্রতিকার": "সাদা মাছি নিয়ন্ত্রণ করুন, আক্রান্ত গাছ সরান।"},
                 {"রোগ": "টমেটো মোজাইক ভাইরাস", "কেন হয়": "ভাইরাস দ্বারা সংক্রমণ।", "প্রতিকার": "রোগমুক্ত বীজ ব্যবহার করুন, গাছ ও পোকা নিয়ন্ত্রণ করুন।"},
                 'টমেটো___সুস্থ']

            prediction = class_name[result_index]
            st.success(f"Model predicts: {prediction['রোগ']}")
            st.info(f"কেন হয়: {prediction['কেন হয়']}")
            st.warning(f"প্রতিকার: {prediction['প্রতিকার']}")

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
