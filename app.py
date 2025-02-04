import streamlit as st
import pytesseract
from PIL import Image, ImageChops, ImageEnhance, ImageDraw
import io
import numpy as np
import cv2  # Import OpenCV for edge detection
import fitz  # PyMuPDF for PDF handling
from pdf2image import convert_from_bytes  # Import pdf2image
import piexif
from PIL.Image import ExifTags  # For EXIF tag lookup

# Layout
st.set_page_config(layout="wide")

# Error Level Analysis (ELA) Function
def detect_bright_spots(image, resave_quality=40, enhancement=30, threshold_value=30):
    # Load the original image
    original_image = image

    # Save at lower quality to introduce JPEG artifacts
    temp_image_path = "temp_resaved.jpg"
    original_image.save(temp_image_path, 'JPEG', quality=resave_quality)

    # Load the resaved image
    resaved_image = Image.open(temp_image_path)

    # Compute the ELA image (difference)
    ela_image = ImageChops.difference(original_image, resaved_image)

    # Enhance brightness for better visualization
    enhancer = ImageEnhance.Brightness(ela_image)
    ela_image = enhancer.enhance(enhancement)

    # Convert ELA image to numpy array for OpenCV processing
    ela_array = np.array(ela_image.convert('RGB'))

    # Convert to grayscale for intensity filtering
    gray_ela = cv2.cvtColor(ela_array, cv2.COLOR_RGB2GRAY)

    # Threshold to capture only bright tampering spots
    _, bright_spots_mask = cv2.threshold(gray_ela, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours of bright tampering regions
    contours, _ = cv2.findContours(bright_spots_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Load the original image in OpenCV format
    original_cv2 = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    for a in contours:
        print(cv2.contourArea(a))
    contours_new = [x for x in contours if cv2.contourArea(x) > 1000 and cv2.contourArea(x) < 2000] #or cv2.contourArea(x) > 4000 ]
    print(contours_new)
    if len(contours_new) == 0:
        # contours_new = [x for x in contours if cv2.contourArea(x) < 1000] #and cv2.contourArea(x) < 2000]
        print(contours_new)

    # Draw contours only for bright spots
    cv2.drawContours(original_cv2, contours_new, -1, (0, 0, 255), 4)  # red for bright tampering areas

    # Convert back to PIL image to display
    ela_image_with_contours = Image.fromarray(cv2.cvtColor(original_cv2, cv2.COLOR_BGR2RGB))

    return ela_image_with_contours


# Metadata extraction using PyMuPDF (fitz)
def extract_metadata_pymupdf(pdf_bytes):
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        metadata = pdf_document.metadata
        pdf_document.close()

        # Check for potential anomalies in metadata
        anomalies = []
        if metadata.get("title") is None:
            anomalies.append("No title found")
        if metadata.get("author") is None:
            anomalies.append("No author found")
        if metadata.get("creationDate") and metadata.get("modDate"):
            creation_date = metadata["creationDate"]
            modification_date = metadata["modDate"]
            if creation_date != modification_date:
                anomalies.append(f"Inconsistent creation and modification dates: {creation_date} vs {modification_date}")
        if metadata.get("producer") == "Unknown" or "Not Available" in metadata.get("producer", ""):
            anomalies.append("Suspicious producer information (Unknown or Not Available)")

        return {"metadata": metadata, "anomalies": anomalies}
    except Exception as e:
        return {"Error": f"Error reading metadata with PyMuPDF: {e}"}


# Convert PDF pages to images
def pdf_to_images(pdf_bytes):
    images = convert_from_bytes(pdf_bytes)
    return images


# Function to extract EXIF data from images
def extract_exif_data(image):
    try:
        exif_data = image._getexif()
        if exif_data:
            exif_info = {}
            for tag, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                exif_info[tag_name] = value
            return exif_info
        else:
            return "No EXIF data found."
    except Exception as e:
        return f"Error extracting EXIF data: {str(e)}"


# Title for the Streamlit app
col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("Upload a JPEG image or PDF", type=["jpg", "jpeg", "pdf"])
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            pdf_bytes = uploaded_file.read()
            images = pdf_to_images(pdf_bytes)
            st.write("**PDF Converted to Images:**")
            for i, img in enumerate(images):
                st.image(img, caption=f"Page {i+1}", use_container_width=True)
        else:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.title("DOCUMENT FORGERY ANALYSIS TOOL FOR SCANNED PDFS")

    # Add color band based on metadata analysis
    if uploaded_file:
        ela_results_all_pages = []
        
        if uploaded_file.type == "application/pdf":
            result = extract_metadata_pymupdf(pdf_bytes)
            metadata = result.get("metadata")
            anomalies = result.get("anomalies")
            
            # Add color band for metadata analysis status
            if anomalies:
                if "Suspicious" in anomalies or "Inconsistent" in anomalies:
                    status = "Critical Anomalies Detected"
                    color = "red"
                else:
                    status = "Some Anomalies Detected"
                    color = "yellow"
            else:
                status = "No Anomalies Detected"
                color = "green"

            # Display colored status band after the title
            st.markdown(
                f"<div style='background-color:{color}; padding: 10px; color: black; font-weight: bold;'>"
                f"{status}</div>",
                unsafe_allow_html=True
            )

            # Display PDF metadata analysis
            st.write("**PDF Metadata:**")
            for key, value in metadata.items():
                st.write(f"{key}: {value}")

            # ELA (Error Level Analysis) for each page
            with st.expander("ERROR LEVEL ANALYSIS (All Pages)"):
                for i, img in enumerate(images):
                    ela_result = detect_bright_spots(img)
                    ela_results_all_pages.append(ela_result)

                for i, ela_result in enumerate(ela_results_all_pages):
                    if ela_result is not None:
                        st.image(ela_result, caption=f"ELA - Page {i+1}", use_container_width=True)

        else:
            ela_result = detect_bright_spots(image)
            if ela_result is not None:
                with st.expander("ERROR LEVEL ANALYSIS (ELA)"):
                    st.image(ela_result, caption="Error Level Analysis", use_container_width=True)

            with st.expander("METADATA ANALYSIS IMAGE"):
                exif_data = extract_exif_data(image)
                if isinstance(exif_data, dict):
                    st.write("**EXIF Metadata:**")
                    for key, value in exif_data.items():
                        st.write(f"{key}: {value}")
                else:
                    st.write(exif_data)