import streamlit as st
import torch
import timm
import gdown
import os
from PIL import Image
import torchvision.transforms as transforms
import pyrebase

# -----------------------------
# 🔐 HIDE STREAMLIT MENU
# -----------------------------
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# 🔥 FIREBASE CONFIG (PUT YOUR KEYS)
# -----------------------------
firebase_config = {
    "apiKey": "YOUR_API_KEY",
    "authDomain": "YOUR_DOMAIN",
    "projectId": "YOUR_PROJECT_ID",
    "storageBucket": "YOUR_BUCKET",
    "messagingSenderId": "XXX",
    "appId": "XXX",
    "databaseURL": ""
}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

# -----------------------------
# SESSION
# -----------------------------
if "user" not in st.session_state:
    st.session_state.user = None

# -----------------------------
# 🔐 LOGIN SYSTEM
# -----------------------------
if not st.session_state.user:
    st.title("🔐 Login System")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns(2)

    if col1.button("Login"):
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            st.session_state.user = user
            st.success("Login successful ✅")
            st.rerun()
        except:
            st.error("Invalid credentials ❌")

    if col2.button("Signup"):
        try:
            auth.create_user_with_email_and_password(email, password)
            st.success("Account created ✅")
        except:
            st.error("Signup failed ❌")

    st.stop()   # ⛔ block app if not logged in

# -----------------------------
# ROLE DETECTION
# -----------------------------
user_email = st.session_state.user["email"]

if user_email == "your_admin_email@gmail.com":
    role = "Admin"
else:
    role = "Patient"

st.sidebar.success(f"Logged in as: {role}")

if st.sidebar.button("Logout"):
    st.session_state.user = None
    st.rerun()

# -----------------------------
# ADMIN DASHBOARD
# -----------------------------
if role == "Admin":
    st.title("👨‍⚕️ Admin Dashboard")

    st.write("✔ Monitor system")
    st.write("✔ View usage")
    st.write("✔ Manage users")

# -----------------------------
# PATIENT PORTAL (MODEL)
# -----------------------------
else:
    st.set_page_config(page_title="DR Detection", layout="centered")

    st.title("👁️ Diabetic Retinopathy Detection")
    st.markdown("### AI-powered retinal analysis")

    # -----------------------------
    # MODEL CONFIG
    # -----------------------------
    MODEL_PATH = "model.pth"
    MODEL_URL = "https://drive.google.com/uc?id=1yDdDELohhVrnI_SSRAQAbqkruoV0fBpw"

    labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

    # -----------------------------
    # DOWNLOAD MODEL
    # -----------------------------
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model... ⏳")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        st.success("Model downloaded ✅")

    # -----------------------------
    # LOAD MODEL
    # -----------------------------
    @st.cache_resource
    def load_model():
        model = timm.create_model('convnext_base', pretrained=False, num_classes=5)
        state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    model = load_model()

    # -----------------------------
    # UI
    # -----------------------------
    st.sidebar.title("About")
    st.sidebar.write("Upload retinal image to detect DR severity")

    file = st.file_uploader("Upload Retina Image", type=["jpg", "png", "jpeg"])

    if file:
        try:
            img = Image.open(file).convert("RGB")
            st.image(img, caption="Uploaded Image", use_column_width=True)

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

            input_tensor = transform(img).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                pred = torch.argmax(probs, dim=1).item()

            st.success(f"Prediction: {labels[pred]}")

            st.subheader("Confidence Scores")
            for i, p in enumerate(probs[0]):
                st.write(f"{labels[i]}: {p.item()*100:.2f}%")

        except Exception as e:
            st.error("Error processing image")
            st.text(str(e))
