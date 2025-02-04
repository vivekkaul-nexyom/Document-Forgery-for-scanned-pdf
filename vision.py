import cv2
import pytesseract
import os
import shutil
import numpy as np
from PIL import Image
import streamlit as st
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF for PDF handling

# Set the page configuration for wider layout
st.set_page_config(page_title="Document Fraud Detection Tool", layout="wide")

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

# Streamlit UI
st.title("Document Fraud Detection Tool")

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

# Adjusted the section to handle tampered vs trusted documents clearly
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

        # Check if the document is tampered or trusted
        if similarity_percentage < 95:
            fraud_count += 1
            marked_image_path = comparator.save_marked_image(claimed_image_cv2, comparator.template, fraud_folder)
            fraud_message = "Document is tampered."  # Updated message
            fraud_messages.append((fraud_message, marked_image_path))
        else:
            genuine_count += 1
            shutil.copy(claimed_path, genuine_folder)
            st.success(f"Document is trusted.")  # Updated message

# Adjust the section that displays the analysis
with col3:
    st.header("Fraudulent Document Analysis")
    st.write(f"Genuine Documents: {genuine_count}")
    st.write(f"Fraudulent Documents: {fraud_count}")

    # Display messages for fraudulent documents
    if fraud_messages:
        st.subheader("Document Analysis")
        for fraud_message, marked_image_path in fraud_messages:
            st.markdown(f"<span style='color:red;font-weight:bold;'>{fraud_message}</span>", unsafe_allow_html=True)
            st.image(marked_image_path, caption=f'Marked Image', use_container_width=True)

    # Display messages for genuine documents
    if genuine_count > 0:
        st.subheader("Document Analysis")
        for file in claimed_files:
            # Check if the document is genuine or tampered
            if file.name in [file.name for fraud_message, _ in fraud_messages]:  # Avoid duplicates
                continue
            st.markdown(f"<span style='color:green;font-weight:bold;'>Document is trusted.</span>", unsafe_allow_html=True)