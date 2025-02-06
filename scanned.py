import streamlit as st
import cv2
import pytesseract
import os
import shutil
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageDraw
import tempfile
import fitz  # PyMuPDF for PDF handling
from pdf2image import convert_from_bytes  # Import pdf2image
from PIL.Image import ExifTags  # For EXIF tag lookup
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Helper function to convert image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Set the page configuration for wider layout
st.set_page_config(page_title="Document Fraud Detection Tool", layout="wide")

# Branding Header with Logo
logo_path = "C:\\8SCANNED PDF PROJECT\\company logo.png"  # Ensure the logo is in the same directory or adjust the path
if os.path.exists(logo_path):
    logo_base64 = image_to_base64(logo_path)
else:
    logo_base64 = None

st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div style="flex: 1; text-align: left; display: flex; align-items: flex-start;">
            {'<img src="data:image/png;base64, ' + logo_base64 + '" style="height: 40px; margin-top: -90px;" alt="Logo">' if logo_base64 else '' }
        </div>
        <div style="flex: 10; text-align: center; margin-top: -30px;">
            <h1 style="font-size: 50px; color: #336699;">
                <span style="color: #4d4d4d;">SCANNED</span> 
                <span style="background: linear-gradient(to right, #1abc9c, #2ecc71); -webkit-background-clip: text; color: transparent;">DOC</span>
                <span style="color: #4d4d4d;">.AI</span>
            </h1>
            <h3 style="color: #2ecc71;">Document Fraud Detection and Analysis Tool</h3>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Radio buttons to choose between the two tools
option = st.radio(
    "Select the tool to use:",
    ("ELA/Metadata analysis", "Document Comparison")
)

# Error Level Analysis (ELA) Function
def error_level_analysis(image, save_path="highlighted_top_errors.jpg", brightness_factor=30, threshold=30, top_n=15):
    try:
        original = image.convert("RGB")
        temp_path = "temp_compressed.jpg"
        original.save(temp_path, "JPEG", quality=80)
        compressed = Image.open(temp_path)
        ela_image = ImageChops.difference(original, compressed)
        ela_image = ImageEnhance.Brightness(ela_image).enhance(brightness_factor)
        ela_array = np.array(ela_image)
        gray_ela = cv2.cvtColor(ela_array, cv2.COLOR_RGB2GRAY)
        _, binary_mask = cv2.threshold(gray_ela, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:top_n:-5]
        highlighted_image = original.copy()
        draw = ImageDraw.Draw(highlighted_image)
        for contour in sorted_contours:
            x, y, w, h = cv2.boundingRect(contour)
            draw.rectangle([x, y, x+w, y+h], outline="red", width=3)
        highlighted_image.save(save_path)
        return ela_array
    except Exception as e:
        return None

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

# Document Comparator Class
class DocumentComparator:
    def __init__(self):
        self.template = None
        self.claimed_documents = []
        self.standard_text_file = "standard_text.txt"

    def load_file(self, file_path):
        return cv2.imread(file_path)

    def extract_text_from_image(self, image):
        if image is None:
            raise ValueError("Image input is empty or invalid.")
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        extracted_text = pytesseract.image_to_string(thresholded, lang='eng', config='--psm 6')
        return extracted_text

    def extract_images_from_pdf(self, pdf_file):
        images = []
        doc = fitz.open(pdf_file)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(image)
        return images

    def detect_fraud(self, extracted_text):
        if self.template is None or not extracted_text:
            return False

        with open(self.standard_text_file, "r") as f:
            standard_text = f.read()

        vectorizer = TfidfVectorizer().fit_transform([extracted_text, standard_text])
        cosine_sim = cosine_similarity(vectorizer[0:1], vectorizer[1:2])
        similarity_percentage = cosine_sim[0][0] * 100

        return similarity_percentage

    def save_marked_image(self, image, template, folder):
        marked_image = image.copy()
        image_resized = cv2.resize(image, (template.shape[1], template.shape[0]))
        difference = cv2.absdiff(template, image_resized)
        _, thresholded = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)
        thresholded_gray = cv2.cvtColor(thresholded, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(thresholded_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(marked_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        marked_image_path = os.path.join(folder, "marked_image.png")
        cv2.imwrite(marked_image_path, marked_image)
        return marked_image_path
    
     # Helper function to extract image metadata
    def extract_image_metadata(image: Image.Image):
     exif_data = image._getexif()  # Extract EXIF metadata from the image
     if not exif_data:
        return {"Metadata": "No EXIF metadata found."}
    
     metadata = {Image.ExifTags.TAGS.get(tag, tag): value for tag, value in exif_data.items() if tag in Image.ExifTags.TAGS}
     return metadata    

    # Streamlit UI Layout for Document Comparison and ELA/Metadata Analysis
    if option == "ELA/Metadata analysis":
        # ELA/Metadata Analysis Section
        col1, col2 = st.columns([1, 2])

        with col1:
            uploaded_file = st.file_uploader("Upload a JPEG image or PDF", type=["jpg", "jpeg", "pdf"])
            if uploaded_file:
                if uploaded_file.type == "application/pdf":
                    pdf_bytes = uploaded_file.read()
                    images = pdf_to_images(pdf_bytes)
                    st.write("**PDF Converted to Images:**")
                    # Only display images in col1, no ELA analysis here
                    for i, img in enumerate(images):
                        st.image(img, caption=f"Page {i+1}", use_container_width=True)
                else:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.title("ANALYSIS")

            # Metadata Extraction (for PDF or Image)
            if uploaded_file:
                if uploaded_file.type == "application/pdf":
                    result = extract_metadata_pymupdf(pdf_bytes)
                    metadata = result.get("metadata")
                    anomalies = result.get("anomalies")
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

                    st.markdown(
                        f"<div style='background-color:{color}; padding: 10px; color: black; font-weight: bold;'>"
                        f"{status}</div>",
                        unsafe_allow_html=True
                    )

                    st.write("**PDF Metadata:**")
                    for key, value in metadata.items():
                        st.write(f"{key}: {value}")

                else:
                    # Metadata Analysis for Image
                    st.write("**Image Metadata:**")
                    image_metadata = extract_image_metadata(image)
                    for key, value in image_metadata.items():
                        st.write(f"{key}: {value}")

                    # ELA for the image
                    result = error_level_analysis(image)
                    if result is not None:
                        with st.expander("ERROR LEVEL ANALYSIS (ELA)"):
                            st.image(result, caption="Error Level Analysis", use_container_width=True)

            # ELA for PDFs
            if uploaded_file:
                if uploaded_file.type == "application/pdf":
                    for img in images:
                        ela_result = error_level_analysis(img)
                        if ela_result is not None:
                            with st.expander("ERROR LEVEL ANALYSIS (ELA)"):
                                st.image(ela_result, caption="Error Level Analysis", use_container_width=True)  


if option == "Document Comparison":
    try:
        comparator = DocumentComparator()

        # Create three columns for a structured layout
        col1, col3, col2 = st.columns([1, 2, 1])

        with col1:
            st.header("Step 1: Upload Standard Document")
            template_file = st.file_uploader("Upload the standard template", type=["png", "jpg", "jpeg", "pdf"])

            if template_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(template_file.name)[1]) as tmp_template:
                    template_path = tmp_template.name
                    tmp_template.write(template_file.read())

                if template_file.type == "application/pdf":
                    images = comparator.extract_images_from_pdf(template_path)
                    if images:
                        template_image = images[0]
                        template_image.save(template_path)
                        comparator.template = cv2.cvtColor(np.array(template_image), cv2.COLOR_RGB2BGR)
                else:
                    template_image = Image.open(template_file)
                    template_image.save(template_path)
                    comparator.template = cv2.imread(template_path)

                standard_text = comparator.extract_text_from_image(comparator.template)
                with open(comparator.standard_text_file, "w") as f:
                    f.write(standard_text)

                st.image(template_image, caption='Uploaded Template Document', use_container_width=True)
                st.success("Template selected successfully.")

        with col2:
            st.header("Step 2: Upload Claimed Document")
            claimed_files = st.file_uploader("Upload claimed document", type=["png", "jpg", "jpeg", "pdf"], accept_multiple_files=True)

            genuine_count = 0
            fraud_count = 0
            genuine_folder = "genuine"
            fraud_folder = "fraud"
            os.makedirs(genuine_folder, exist_ok=True)
            os.makedirs(fraud_folder, exist_ok=True)

            fraud_messages = []

            for file in claimed_files:
                file_extension = os.path.splitext(file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_claimed:
                    claimed_path = tmp_claimed.name
                    tmp_claimed.write(file.read())

                if file.type == "application/pdf":
                    images = comparator.extract_images_from_pdf(claimed_path)
                    if not images:
                        st.warning(f"No images found in PDF '{file.name}'")
                        continue
                else:
                    images = [Image.open(claimed_path)]

                for idx, claimed_image in enumerate(images):
                    st.image(claimed_image, caption=f'Uploaded Document: {file.name} (Page {idx + 1})', use_container_width=True)

                claimed_image_np = np.array(claimed_image)
                claimed_image_cv2 = cv2.cvtColor(claimed_image_np, cv2.COLOR_RGB2BGR)

                extracted_text = comparator.extract_text_from_image(claimed_image_cv2)
                similarity_percentage = comparator.detect_fraud(extracted_text)

                if similarity_percentage < 95:
                    fraud_count += 1
                    marked_image_path = comparator.save_marked_image(claimed_image_cv2, comparator.template, fraud_folder)
                    fraud_message = "Document is tampered."
                    fraud_messages.append((fraud_message, marked_image_path))
                else:
                    genuine_count += 1
                    shutil.copy(claimed_path, genuine_folder)
                    st.success(f"Document is trusted.")

        with col3:
            st.header("Fraudulent Document Analysis")
            st.write(f"Genuine Documents: {genuine_count}")
            st.write(f"Fraudulent Documents: {fraud_count}")

            if fraud_messages:
                st.subheader("Document Analysis")
                for fraud_message, marked_image_path in fraud_messages:
                    st.markdown(f"<span style='color:red;font-weight:bold;'>{fraud_message}</span>", unsafe_allow_html=True)
                    st.image(marked_image_path, caption=f'Marked Image', use_container_width=True)

            if genuine_count > 0:
                st.subheader("Document Analysis")
                for file in claimed_files:
                    if file.name in [file.name for fraud_message, _ in fraud_messages]:
                        continue
                    st.markdown(f"<span style='color:green;font-weight:bold;'>Document is trusted.</span>", unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")