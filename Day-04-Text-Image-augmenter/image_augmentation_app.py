import streamlit as st
from PIL import Image, ImageEnhance
import io
import zipfile
import os

st.title("Image Augmentation Gallery")

# File uploader
uploaded_files = st.file_uploader("Upload images", type=["jpg", "png"], accept_multiple_files=True)

# Augmentation options
resize = st.checkbox("Resize")
if resize:
    width = st.number_input("Width", min_value=1, value=200)
    height = st.number_input("Height", min_value=1, value=200)
rotate = st.checkbox("Rotate")
if rotate:
    angle = st.slider("Rotation Angle", -180, 180, 0)
flip = st.checkbox("Flip")
brightness = st.checkbox("Adjust Brightness")
if brightness:
    factor = st.slider("Brightness Factor", 0.1, 2.0, 1.0)

if uploaded_files and st.button("Augment Images"):
    augmented_images = []
    for i, file in enumerate(uploaded_files):
        image = Image.open(file)
        processed = image.copy()
        
        if resize:
            processed = processed.resize((width, height))
        if rotate:
            processed = processed.rotate(angle)
        if flip:
            processed = processed.transpose(Image.FLIP_LEFT_RIGHT)
        if brightness:
            processed = ImageEnhance.Brightness(processed).enhance(factor)
        
        augmented_images.append((f"augmented_{i}.png", processed))
        
        # Display
        st.image(processed, caption=f"Augmented Image {i+1}", width=200)
    
    # Create zip
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for name, img in augmented_images:
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="PNG")
            zip_file.writestr(name, img_buffer.getvalue())
    
    st.download_button(
        label="Download Augmented Images (ZIP)",
        data=zip_buffer.getvalue(),
        file_name="augmented_images.zip",
        mime="application/zip"
    )