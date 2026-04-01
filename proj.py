import streamlit as st
import torch
import timm
import requests
import os
from PIL import Image
import torchvision.transforms as transforms

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "model.pth"
MODEL_URL = "https://drive.google.com/uc?id=1yDdDELohhVrnI_SSRAQAbqkruoV0fBpw"

labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# -----------------------------
# Download model
# -----------------------------
def download_model(url, output_path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

# Download if not exists
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model... (first time only ⏳)")
    download_model(MODEL_URL, MODEL_PATH)
    st.success("Model downloaded ✅")

# -----------------------------
# Load model (FIXED)
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
st.title("👁️ Diabetic Retinopathy Detection")

file = st.file_uploader("Upload Retina Image", type=["jpg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img)

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

    for i, p in enumerate(probs[0]):
        st.write(f"{labels[i]}: {p.item()*100:.2f}%")
