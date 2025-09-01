# cifar10_resnet_streamlit.py
# streamlit_imagenet_classifier.py
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
import json
import requests

# ----------------------------
# Load ImageNet class labels
# ----------------------------
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = requests.get(LABELS_URL).json()  # 1000 ImageNet classes

# ----------------------------
# Load pretrained ResNet18
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model = model.to(device)
model.eval()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🖼️ Real-World Image Classifier")
st.markdown("Upload any image and the model will predict its class!")

uploaded_file = st.file_uploader("Choose an image...", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image for ResNet
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        top_prob, top_idx = torch.max(probabilities, 1)
        predicted_class = labels[top_idx.item()]

    # Display predicted class
    st.success(f"Predicted class: **{predicted_class}** (Confidence: {top_prob.item()*100:.2f}%)")
