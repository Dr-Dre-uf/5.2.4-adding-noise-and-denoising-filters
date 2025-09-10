import streamlit as st
import numpy as np
import cv2
from PIL import Image
from skimage.util import random_noise
from skimage import img_as_ubyte

st.set_page_config(page_title="Salt & Pepper Noise and Denoising", layout="wide")

st.title("Adding Salt-and-Pepper Noise and Applying Denoising Filters")

# Warning
st.warning("⚠️ Do not upload sensitive or personal data. Images are processed locally in this demo app.")

# Sidebar: image selection
st.sidebar.header("Image Selection")
use_uploaded = st.sidebar.checkbox("Upload your own image")

uploaded_img = None
if use_uploaded:
    uploaded_img = st.sidebar.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )
else:
    image_choice = st.sidebar.selectbox(
        "Select Example Image",
        ["Fluorescence (IFCells)", "Brightfield (BloodSmear)"],
        help="Choose a sample image."
    )

# Sidebar: noise and filter settings
st.sidebar.header("Noise Settings")
noise_amount = st.sidebar.slider(
    "Noise Amount", 0.0, 0.2, 0.05, step=0.01,
    help="Controls how much salt-and-pepper noise is added."
)

st.sidebar.header("Denoising Filter")
filter_type = st.sidebar.radio(
    "Choose Filter Type", ["Median", "Gaussian"],
    help="Select the type of filter used to remove noise."
)

if filter_type == "Median":
    filter_strength = st.sidebar.slider(
        "Kernel Size (odd only)", 3, 11, 3, step=2,
        help="Size of the kernel for the median filter."
    )
else:
    filter_strength = st.sidebar.slider(
        "Gaussian Sigma", 0.5, 5.0, 1.0, step=0.5,
        help="Standard deviation for Gaussian smoothing."
    )

# Default paths
bf_path = "assets/BloodSmear.png"
if_path = "assets/IFCells.jpg"

# Load image (color-safe)
if use_uploaded and uploaded_img is not None:
    img = np.array(Image.open(uploaded_img).convert("RGB"))
elif not use_uploaded:
    if image_choice == "Fluorescence (IFCells)":
        img = np.array(Image.open(if_path).convert("RGB"))
    else:
        img = np.array(Image.open(bf_path).convert("RGB"))
else:
    img = None

# Process
if img is not None:
    # Add salt & pepper noise (skimage gives float [0,1])
    noisy_img = random_noise(img, mode="s&p", amount=noise_amount)
    noisy_img_u8 = img_as_ubyte(noisy_img)

    # Apply selected filter
    if filter_type == "Median":
        denoised_img = cv2.medianBlur(noisy_img_u8, filter_strength)
    else:  # Gaussian
        denoised_img = cv2.GaussianBlur(noisy_img_u8, (5, 5), filter_strength)

    # Display results
    st.subheader("Noise Simulation and Denoising")
    col1, col2, col3 = st.columns(3)
    col1.image(img, caption="Original", use_container_width=True)
    col2.image(noisy_img, caption="With Salt & Pepper Noise", use_container_width=True)
    col3.image(denoised_img, caption=f"Denoised ({filter_type})", use_container_width=True)
else:
    st.info("Please upload an image or select one from the examples to begin.")
