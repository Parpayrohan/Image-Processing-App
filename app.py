import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from io import BytesIO

# -----------------------------
# Load DeepLabV3 model for segmentation
# -----------------------------
@st.cache(allow_output_mutation=True)
def load_model():
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.eval()
    return model

model = load_model()

# Transformation for segmentation model
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("AI-Based Image Processing")

# Sidebar
theme_option = st.sidebar.radio("Select Theme", ("Light", "Dark"))
option = st.sidebar.selectbox("Select Operation", 
    ("None", "Segmentation", "GrabCut", "Denoising", "Deblurring", "Super Resolution", "Cartoonize"))

# Apply theme
if theme_option == "Dark":
    st.markdown("""
        <style>
        body { background-color:#2e2e2e; color:white; }
        .sidebar .sidebar-content { background-color:#2e2e2e; color:white; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body { background-color:white; color:black; }
        .sidebar .sidebar-content { background-color:white; color:black; }
        </style>
    """, unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Convert image to bytes for download
def image_to_bytes(image):
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

# -----------------------------
# Process uploaded image
# -----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("Uploaded Image")
    st.image(image, use_column_width=True)

    processed_image = None

    if option == "Segmentation":
        st.write("Running DeepLabV3 Segmentation...")
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(image_tensor)['out'][0]
        mask = torch.argmax(output, dim=0).cpu().numpy()

        # Visualization
        plt.figure(figsize=(6,6))
        plt.imshow(mask, cmap='jet')
        plt.axis('off')
        st.pyplot(plt)

        processed_image = Image.fromarray(mask.astype(np.uint8))

    elif option == "GrabCut":
        st.write("Running GrabCut segmentation...")
        img_cv = np.array(image)
        mask = np.zeros(img_cv.shape[:2], np.uint8)
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)
        rect = (50,50,img_cv.shape[1]-100,img_cv.shape[0]-100)
        cv2.grabCut(img_cv, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        output = img_cv * mask2[:,:,np.newaxis]
        st.image(output, use_column_width=True)
        processed_image = Image.fromarray(output)

    elif option == "Denoising":
        st.write("Applying Denoising...")
        img_cv = np.array(image)
        denoised = cv2.fastNlMeansDenoisingColored(img_cv,None,10,10,7,21)
        st.image(denoised, use_column_width=True)
        processed_image = Image.fromarray(denoised)

    elif option == "Deblurring":
        st.write("Applying Motion Blur Deblurring...")
        img_cv = np.array(image)
        kernel = np.zeros((5,5))
        kernel[2,:] = np.ones(5)/5
        deblurred = cv2.filter2D(img_cv, -1, kernel)
        st.image(deblurred, use_column_width=True)
        processed_image = Image.fromarray(deblurred)

    elif option == "Super Resolution":
        st.write("Super Resolution function not implemented yet.")
        st.image(image, use_column_width=True)
        processed_image = image

    elif option == "Cartoonize":
        st.write("Cartoonizing Image...")
        img_cv = np.array(image)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(img_cv, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        st.image(cartoon, use_column_width=True)
        processed_image = Image.fromarray(cartoon)

    # Download button
    if processed_image is not None:
        st.download_button(
            label="⬇️ Download Processed Image",
            data=image_to_bytes(processed_image),
            file_name="processed_image.png",
            mime="image/png"
        )
