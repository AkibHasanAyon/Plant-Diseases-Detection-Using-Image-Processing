# 🌿 Plant Diseases Detection Using Image Processing & Deep Learning

![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Android](https://img.shields.io/badge/Android-Java%20%2F%20Kotlin-3DDC84?style=for-the-badge&logo=android&logoColor=white)

An end-to-end intelligent agro-diagnostic system combining Convolutional Neural Networks (CNNs), a Streamlit web portal, and a native Android mobile application to identify **38 distinct plant leaf diseases** and provide actionable agricultural remedies in Bengali.

---

## 🌟 Key Highlights

* **38 Disease Classes Detected**: Classifies diseases across major crops including Apple, Tomato, Potato, Corn, Grape, Strawberry, Peach, Cherry, Pepper, and Sweet Potato.
* **Bilingual Farmer Recommendations**: Predictions provide the disease identity, causal root causes (*কেন হয়*), and practical agricultural cures (*প্রতিকার*) in Bengali.
* **Dual-Platform Deployment**:
  * **Web Portal (`admin.py`)**: Streamlit application with role-based access for farmers and agricultural administrators.
  * **Mobile App (`mobile app/`)**: Native Android application enabling on-the-field diagnosis without needing desktop hardware.
* **Input Audit Logging**: Logs predictions, farmer contact numbers, and uploaded specimen photos to [`user_inputs.json`](file:///C:/Users/HP/Downloads/Ayon/Plant-Diseases-Detection-Using-Image-Processing/user_inputs.json) for historical advisory tracking.

---

## 🏗️ Project Architecture

```text
Plant-Diseases-Detection-Using-Image-Processing/
├── admin.py                           # Streamlit web application & prediction interface
├── user_reg.py                        # User registration and authentication helpers
├── requirements.txt                   # Python package dependencies
├── home_page.jpeg                     # Portal UI banner asset
├── user_data.json                     # Registered users and credentials database
├── user_inputs.json                   # Prediction audit logs (mobile, image, timestamp)
├── uploaded_images/                   # Stored farmer leaf uploads
│
├── trained_plant_disease_model.h5     # Keras HDF5 trained CNN model weights (94 MB)
├── trained_plant_disease_model.keras  # Keras v3 trained model format (94 MB)
├── training_hist.json                 # Epoch-by-epoch loss & accuracy training log
├── Train_plant_disease.ipynb          # CNN model architecture & training pipeline
├── Test_plant_disease.ipynb           # Model evaluation & confusion matrix
│
├── mobile app/                        # Native Android Studio Project
│   ├── app/src/main/                  # Java/Kotlin activities, layouts, manifests
│   └── gradle/                        # Android build system files
│
└── Reports/                           # Comprehensive Engineering & Course Reports
    ├── CO1 - Project Proposal.docx
    ├── CO2 - Design Report.docx
    ├── CO2 - Project Impact and Sustainability Report.docx
    ├── CO3 - Modern Tool Usage Report.docx
    ├── CO4 - Project Impact and Constraints Report.docx
    └── CO5 - Final Report.docx
```

---

## 🧠 Deep Learning Architecture & Classes

The model uses a deep Convolutional Neural Network trained on resized leaf RGB images ($128 \times 128 \times 3$).

### Supported Crops & Conditions:
* **Apple**: Apple Scab, Black Rot, Cedar Apple Rust, Healthy
* **Blueberry**: Healthy
* **Cherry**: Powdery Mildew, Healthy
* **Corn (Maize)**: Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy
* **Grape**: Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis), Healthy
* **Orange**: Citrus Greening (Huanglongbing)
* **Peach**: Bacterial Spot, Healthy
* **Pepper (Bell)**: Bacterial Spot, Healthy
* **Potato**: Early Blight, Late Blight, Healthy
* **Raspberry**: Healthy
* **Soybean**: Healthy
* **Squash**: Powdery Mildew
* **Strawberry**: Leaf Scorch, Healthy
* **Tomato**: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy

---

## 🚀 Web Application Setup (Streamlit)

### 1. Prerequisites
* Python 3.9 or 3.10
* Virtual environment tool

### 2. Installation
```bash
# Navigate to directory
cd Plant-Diseases-Detection-Using-Image-Processing

# Create and activate environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required packages
pip install streamlit tensorflow numpy Pillow
```

### 3. Run the Web Application
```bash
streamlit run admin.py
```
Open your browser at `http://localhost:8501`.

### 4. Default Credentials
* **Admin Login**: Username: `admin` | Password: `admin`
* **Farmer Account**: Register a new account via the **Register** tab in the sidebar navigation.

---

## 📱 Android Mobile Application Setup

1. Open **Android Studio**.
2. Select **Open an Existing Project** and choose the [`mobile app`](file:///C:/Users/HP/Downloads/Ayon/Plant-Diseases-Detection-Using-Image-Processing/mobile%20app/) folder.
3. Allow Gradle to synchronize dependencies.
4. Connect an Android device (with USB debugging enabled) or start an Android Emulator.
5. Click **Run 'app'** (`Shift + F10`) to build and deploy the APK.

---

## 📑 Project Reports & Documentation
Refer to the [`Reports/`](file:///C:/Users/HP/Downloads/Ayon/Plant-Diseases-Detection-Using-Image-Processing/Reports/) folder for comprehensive academic evaluations, sustainability analysis, and modern engineering tool matrices.

