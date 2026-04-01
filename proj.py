import streamlit as st
import torch
import timm
from PIL import Image
import torchvision.transforms as transforms

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "best_convnext_fold0.pth"

labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# -----------------------------
# Load model (FIXED for PyTorch 2.6+)
# -----------------------------
@st.cache_resource
def load_model():
    model = timm.create_model('convnext_base', pretrained=False, num_classes=5)
    
    # 🔥 FIX HERE
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    
    model.eval()
    return model

model = load_model()

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="DR Detection", layout="centered")

st.title("👁️ Diabetic Retinopathy Detection")

st.sidebar.title("About")
st.sidebar.write("Upload a retinal image to detect DR severity using AI")

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

        # Prediction
        st.success(f"Prediction: {labels[pred]}")

        # Confidence scores
        st.subheader("Confidence Scores")
        for i, p in enumerate(probs[0]):
            st.write(f"{labels[i]}: {p.item()*100:.2f}%")

    except Exception as e:
        st.error("Error processing image")
        st.text(str(e))
