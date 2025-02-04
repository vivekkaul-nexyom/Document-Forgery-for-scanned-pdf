import streamlit as st
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize
from datetime import datetime
import fitz  # PyMuPDF
import numpy as np
import tempfile
import os
import nltk
import html

# Set up Streamlit page configuration
st.set_page_config(page_title="PDF Forgery Detection Tool", layout="wide")

# Download necessary NLTK resources
nltk.download("punkt")

# Load pre-trained NLP model for embeddings
nlp_model = pipeline("feature-extraction", model="bert-base-uncased", tokenizer="bert-base-uncased")

css = """
<style>
/* General Reset */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: Arial, sans-serif;
    color: #333;
    background-color: #f8f9fa;
}

.container {
    display: flex;
    flex-direction: row;
    max-width: 1200px;
    margin: 20px auto;
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 8px;
    overflow: hidden;
}

/* Left Panel */
.left-panel {
    width: 85%;
    background: #f5f5f5;
    padding: 20px;
    border-right: 1px solid #ddd;
}

.header {
    text-align: center;
    margin-bottom: 20px;
    color: #3c5e39;
}

.title {
    font-size: 1.5em;
    font-weight: bold;
    color: #3c5e39;
}

.ai {
    color: #141414;
}

.risk-level {
    color: white;
    padding: 10px;
    text-align: center;
    border-radius: 4px;
    font-weight: bold;
}

.details p {
    margin-bottom: 10px;
    font-size: 0.9em;
}

.indicators h2 {
    margin: 20px 0 10px;
    font-size: 1.1em;
}

.indicators ul {
    list-style: none;
    padding-left: 0;
}

.indicators li {
    margin: 5px 0;
    padding: 5px;
    border-bottom: 1px solid #ddd;
}

/* Right Panel */
.right-panel {
    width: 65%;
    padding: 20px;
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
    border-left: 1px solid #ddd;
}

.upload-instructions {
    text-align: center;
    margin-bottom: 20px;
    font-size: 1.1em;
    color: #666;
}

.pdf-viewer {
    width: 100%;
    max-height: 800px;
    overflow: auto;
    border: 1px solid #ddd;
    border-radius: 8px;
}
</style>
"""

# Inject custom CSS into Streamlit app
st.markdown(css, unsafe_allow_html=True)


# Function to parse PDF date format
def parse_pdf_date(date_str):
    if date_str.startswith("D:"):
        date_str = date_str[2:]
    date_str = date_str.split("+")[0].split("-")[0]
    formats = ["%Y%m%d%H%M%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


# Function to assess metadata risk with explanations
def assess_metadata_risk_with_explanation(metadata):
    risk_details = []
    creation_date = parse_pdf_date(metadata.get("CreationDate", ""))
    modification_date = parse_pdf_date(metadata.get("ModDate", ""))

    # Check for modification before creation
    if creation_date and modification_date:
        if modification_date < creation_date:
            risk_details.append({"risk": "Red", "reason": "Modification date is earlier than creation date."})
    else:
        if not creation_date:
            risk_details.append({"risk": "Green", "reason": "Creation date is missing or invalid."})
        if not modification_date:
            risk_details.append({"risk": "Green", "reason": "Modification date is missing or invalid."})

    # Add more metadata checks
    if "Producer" not in metadata or not metadata["Producer"]:
        risk_details.append({"risk": "Yellow", "reason": "PDF producer information is missing."})
    if "Creator" not in metadata or not metadata["Creator"]:
        risk_details.append({"risk": "Yellow", "reason": "Document creator information is missing."})

    # Default to Green if no issues are detected
    if not risk_details:
        risk_details.append({"risk": "Green", "reason": "✅ No anomalies detected in the metadata."})

    return risk_details


# Function to extract metadata using PyPDF2
def extract_metadata_pypdf2(pdf_file):
    reader = PdfReader(pdf_file)
    metadata = reader.metadata
    return {key.strip("/"): str(value) for key, value in metadata.items()} if metadata else {}


# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_data = [{"page": i + 1, "text": page.get_text("text")} for i, page in enumerate(doc)]
    return text_data


# Function to analyze text similarity
def analyze_text_similarity(text_data):
    suspicious_sentences = []
    for page in text_data:
        sentences = sent_tokenize(page["text"])
        embeddings = np.array([np.mean(nlp_model(sentence)[0], axis=0) for sentence in sentences])
        similarities = cosine_similarity(embeddings)
        avg_similarities = similarities.mean(axis=1)
        threshold = np.percentile(avg_similarities, 30)
        for i, score in enumerate(avg_similarities):
            if score < threshold:
                risk = "Red" if score < 0.3 else "Yellow" if score < 0.9 else "Green"
                suspicious_sentences.append({"page": page["page"], "sentence": sentences[i], "score": score, "risk": risk})
    return suspicious_sentences


def detect_fonts_and_sizes(pdf_path):
    doc = fitz.open(pdf_path)
    font_details = []

    # Define font size thresholds for risk levels
    small_threshold = 8  # Adjust as needed
    large_threshold = 14  # Adjust as needed

    # Iterate through each page
    for page_num in range(len(doc)):
        page = doc[page_num]
        text_instances = page.get_text("dict")["blocks"]

        # Extract font information from each text block
        for block in text_instances:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        font_name = span["font"]
                        font_size = span["size"]
                        text = span["text"]

                        # Determine risk level based on font size
                        if font_size < small_threshold:
                            risk = "Red"
                        elif small_threshold <= font_size < large_threshold:
                            risk = "Yellow"
                        else:
                            risk = "Green"

                        font_details.append({
                            "page": page_num + 1,
                            "font_name": font_name,
                            "font_size": font_size,
                            "text_snippet": text[:30],  # Limit to 30 chars
                            "risk": risk  # Add risk level
                        })
    return font_details


def detect_unusual_text_properties(pdf_path):
    doc = fitz.open(pdf_path)
    unusual_text_details = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if "lines" in block:
                previous_char = None
                previous_spacing = None

                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"]
                        font_name = span["font"]
                        font_size = span["size"]
                        color = span["color"]

                        # Check for unusual kerning or spacing
                        for i, char in enumerate(text):
                            if previous_char is not None:
                                char_spacing = span["bbox"][0] - previous_char["bbox"][2]
                                if previous_spacing is not None and abs(char_spacing - previous_spacing) > 1.5:
                                    # Assign risk based on the severity of the spacing issue
                                    if abs(char_spacing - previous_spacing) > 3:
                                        risk = "Red"
                                    elif abs(char_spacing - previous_spacing) > 1.5:
                                        risk = "Yellow"
                                    else:
                                        risk = "Green"

                                    unusual_text_details.append({
                                        "page": page_num + 1,
                                        "font_name": font_name,
                                        "font_size": font_size,
                                        "char_spacing": char_spacing,
                                        "text_snippet": text[max(0, i-10):i+10],  # Context window
                                        "risk": risk
                                    })

                                previous_spacing = char_spacing

                            previous_char = span

                        # Check for style inconsistencies within the same block
                        if len(line["spans"]) > 1:
                            for other_span in line["spans"]:
                                if (other_span["font"] != font_name or 
                                    other_span["size"] != font_size or
                                    other_span["color"] != color):
                                    # Assign risk based on the degree of inconsistency
                                    if font_name != other_span["font"] or font_size != other_span["size"]:
                                        risk = "Red"
                                    elif color != other_span["color"]:
                                        risk = "Yellow"
                                    else:
                                        risk = "Green"

                                    unusual_text_details.append({
                                        "page": page_num + 1,
                                        "text_snippet": text[:30],
                                        "issue": "Inconsistent font, size, or color",
                                        "details": {
                                            "current_font": font_name,
                                            "current_size": font_size,
                                            "current_color": color,
                                            "other_font": other_span["font"],
                                            "other_size": other_span["size"],
                                            "other_color": other_span["color"],
                                        },
                                        "risk": risk
                                    })

    return unusual_text_details

def detect_redacted_content(pdf_path):
    doc = fitz.open(pdf_path)
    redacted_content = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        shapes = page.get_drawings()

        for shape in shapes:
            bbox = shape["rect"]
            if shape["fill"] == (0, 0, 0):  # Check for black-filled shapes
                # Check for text within the bounding box
                text_in_box = page.get_textbox(bbox)
                if text_in_box.strip():
                    # Assign risk based on the size of the redacted area
                    redacted_area = bbox.get_area()
                    if redacted_area > 500:  # Large redaction area
                        risk = "Red"
                    elif redacted_area > 200:  # Moderate redaction area
                        risk = "Yellow"
                    else:  # Small redaction area
                        risk = "Green"

                    redacted_content.append({
                        "page": page_num + 1,
                        "bbox": bbox,
                        "hidden_text": text_in_box,
                        "risk": risk
                    })

    return redacted_content


def detect_duplicate_text(pdf_path):
    doc = fitz.open(pdf_path)
    duplicate_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        
        text_seen = {}
        
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text:
                            # Track text occurrences
                            if text in text_seen:
                                text_seen[text]["count"] += 1
                                text_seen[text]["pages"].add(page_num + 1)
                            else:
                                text_seen[text] = {"count": 1, "pages": {page_num + 1}}

        # Identify and assign risk levels to duplicate text
        for text, info in text_seen.items():
            if info["count"] > 3:  # If the text appears more than 3 times, consider high risk
                risk = "Red"
            elif info["count"] == 2:  # Moderate duplication (appears twice)
                risk = "Yellow"
            else:
                risk = "Green"  # Minor or no duplication

            if risk != "Green":  # Only flag duplicates with non-green risk
                duplicate_text.append({
                    "text": text,
                    "count": info["count"],
                    "pages": list(info["pages"]),
                    "risk": risk
                })

    return duplicate_text



def detect_hidden_layers(pdf_path):
    doc = fitz.open(pdf_path)
    hidden_layers = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Collect all visible elements: text, images, shapes
        elements = []

        # Add text elements
        text_blocks = page.get_text("dict")["blocks"]
        for block in text_blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        elements.append({
                            "type": "text",
                            "bbox": tuple(span["bbox"]),
                            "content": span["text"]
                        })

        # Add shapes (e.g., rectangles, circles, etc.)
        shapes = page.get_drawings()
        for shape in shapes:
            elements.append({
                "type": "shape",
                "bbox": tuple(shape["rect"]),
                "content": None
            })

        # Check for overlapping elements
        for i, elem1 in enumerate(elements):
            overlap_count = 0
            for j, elem2 in enumerate(elements):
                if i != j:
                    bbox1 = fitz.Rect(*elem1["bbox"])
                    bbox2 = fitz.Rect(*elem2["bbox"])

                    # If bounding boxes overlap
                    if bbox1.intersects(bbox2):
                        overlap_count += 1

            # If more than 2 elements overlap, assign a "Red" risk level
            if overlap_count > 2:
                hidden_layers.append({
                    "page": page_num + 1,
                    "element_1": elem1,
                    "overlap_count": overlap_count,
                    "risk": "Red"
                })
            else:
                # If overlap is not more than 2, you can assign other risk levels based on overlap percentage
                for j, elem2 in enumerate(elements):
                    if i != j:
                        bbox1 = fitz.Rect(*elem1["bbox"])
                        bbox2 = fitz.Rect(*elem2["bbox"])

                        # If bounding boxes overlap
                        if bbox1.intersects(bbox2):
                            # Calculate the overlap area
                            intersection = bbox1 & bbox2
                            overlap_area = intersection.get_area()
                            bbox1_area = bbox1.get_area()
                            bbox2_area = bbox2.get_area()
                            max_area = max(bbox1_area, bbox2_area)

                            # Calculate overlap percentage
                            overlap_percentage = (overlap_area / max_area) * 100

                            # Assign risk levels based on overlap percentage
                            if overlap_percentage > 20:
                                risk = "Red"
                            elif 10 < overlap_percentage <= 20:
                                risk = "Yellow"
                            else:
                                risk = "Red"

                            hidden_layers.append({
                                "page": page_num + 1,
                                "element_1": elem1,
                                "element_2": elem2,
                                "overlap_percentage": overlap_percentage,
                                "risk": risk
                            })

    return hidden_layers


def detect_watermark_tampering(pdf_path):
    # Open the PDF
    doc = fitz.open(pdf_path)
    issues = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_issues = []

        # Check for transparency in images (often used for watermarks)
        for img in page.get_images(full=True):
            xref = img[0]
            image = doc.extract_image(xref)
            if "smask" in image:  # Soft mask (transparency)
                page_issues.append({
                    "type": "watermark transparency",
                    "risk": "Red",  # High risk for tampering
                    "message": f"Possible watermark tampering detected on page {page_num + 1}. Image contains transparency."
                })

        # Look for potential tampering by searching for specific hidden or modified layers
        display_items = page.get_text("dict")["blocks"]
        
        if len(display_items) > 1:  # Multiple blocks might indicate layered content
            page_issues.append({
                "type": "layered content",
                "risk": "Yellow",  # Moderate risk for overlapping content
                "message": f"Possible hidden or overlapping content detected on page {page_num + 1}."
            })

        # Detecting oddities in text elements, such as inconsistent font size or appearance
        for block in display_items:
            if "spans" in block:
                for span in block["spans"]:
                    # Check for unusual font size or characteristics that suggest tampering
                    if span["size"] < 8 or span["size"] > 50:
                        page_issues.append({
                            "type": "unusual font size",
                            "risk": "Yellow" if span["size"] < 8 else "Red",  # Small fonts are moderate risk, large fonts are high risk
                            "message": f"Unusual font size detected on page {page_num + 1}: {span['text']}"
                        })

        # If any issues were found for this page, add them to the final list
        if page_issues:
            issues.extend(page_issues)

    return issues

    

# Function to determine overall risk
# def determine_overall_risk(metadata_risk, suspicious_sentences, ):
#     if metadata_risk == "Red" or any(s["risk"] == "Red" for s in suspicious_sentences):
#         return "Highly Forged"
#     elif metadata_risk == "Yellow" or any(s["risk"] == "Yellow" for s in suspicious_sentences):
#         return "Moderately Forged"
#     else:
#         return "Trusted"
def determine_overall_risk(metadata_risk, suspicious_sentences, font_details, unusual_text_details,
                            redacted_content, duplicate_text, hidden_layers, issues):
    
    # Initialize overall risk level
    overall_risk = "Trusted"
    
    # Check metadata risk
    if metadata_risk == "Red":
        overall_risk = "Highly Forged"
    elif metadata_risk == "Yellow" and overall_risk != "Highly Forged":
        overall_risk = "Moderately Forged"

    # Check suspicious sentences in the text
    if any(s["risk"] == "Red" for s in suspicious_sentences):
        overall_risk = "Highly Forged"
    elif any(s["risk"] == "Yellow" for s in suspicious_sentences) and overall_risk != "Highly Forged":
        overall_risk = "Moderately Forged"
    
     # Extend this check to font details, unusual text, etc.
    if any(f["risk"] == "Red" for f in font_details):
        overall_risk = "Highly Forged"
    elif any(f["risk"] == "Yellow" for f in font_details) and overall_risk != "Highly Forged":
        overall_risk = "Moderately Forged"    
        
    if any(u["risk"] == "Red" for u in unusual_text_details):
        overall_risk = "Highly Forged"
    elif any(u["risk"] == "Yellow" for u in unusual_text_details) and overall_risk != "Highly Forged":
        overall_risk = "Moderately Forged"
    
    # Check for redacted content
    if redacted_content:
        overall_risk = "Highly Forged"
    
    # Check for duplicate text in the document
    if any(d["risk"] == "Red" for d in duplicate_text):
        overall_risk = "Highly Forged"
    elif any(d["risk"] == "Yellow" for d in duplicate_text) and overall_risk != "Highly Forged":
        overall_risk = "Moderately Forged"
    
    # Check for hidden layers or invisible content
    if any(h["risk"] == "Red" for h in hidden_layers):
        overall_risk = "Highly Forged"
    elif any(h["risk"] == "Yellow" for h in hidden_layers) and overall_risk != "Highly Forged":
        overall_risk = "Moderately Forged"
    
    # Check for any other issues
    if any(issue["risk"] == "Red" for issue in issues):
        overall_risk = "Highly Forged"
    elif any(issue["risk"] == "Yellow" for issue in issues) and overall_risk != "Highly Forged":
        overall_risk = "Moderately Forged"    

    return overall_risk
    
    
color_dict = {"Red": (1, 0, 0), "Yellow": (1, 1, 0), "Green": (0, 1, 0)}


# Function to render highlighted PDF
def render_highlighted_pdf(pdf_path, suspicious_sentences):
    doc = fitz.open(pdf_path)
    for item in suspicious_sentences:
        page = doc[item["page"] - 1]
        for instance in page.search_for(item["sentence"]):
            highlight = page.add_highlight_annot(instance)
            highlight.set_colors({"stroke": color_dict[item["risk"]]})  # color highlight
            highlight.update()
    output_path = "highlighted_output.pdf"
    doc.save(output_path)
    return output_path


# Function to generate HTML for a color-coded risk band
def generate_risk_band(risk_level):
    colors = {
        "Trusted": "#28a745",  # Green
        "Moderately Forged": "#f2ee9b",  # Yellow
        "Highly Forged": "#dc3545",  # Red
    }
    descriptions = {
        "Trusted": "✅ Trusted: No significant anomalies detected.",
        "Moderately Forged": "⚠️ Moderate Risk: Potential anomalies detected.",
        "Highly Forged": "🚨 High Risk: Significant anomalies detected.",
    }
    color = colors[risk_level]
    description = descriptions[risk_level]

    return color, description


# Streamlit app

st.markdown("""
    <div style="text-align: center; border-top: 1px solid #006400; width: 100%; margin: 0 auto; padding-top: 5px;">
        <span style="font-size: 18px; color: #006400; font-weight: bold;">nexyom.ai</span>
    </div>
""", unsafe_allow_html=True)

st.title("PDF Forgery Detection Tool")
# st.write("Upload a PDF to analyze its forgery.")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file.read())
        temp_path = temp_pdf.name

    # Extract metadata and assess risk
    metadata = extract_metadata_pypdf2(temp_path)
    metadata_risk_details = assess_metadata_risk_with_explanation(metadata)

    # Extract text and analyze suspicious sentences
    text_data = extract_text_from_pdf(temp_path)
    suspicious_sentences = analyze_text_similarity(text_data)

    font_details = detect_fonts_and_sizes(temp_path)  # Initialize font_details
    
    # Detect unusual text properties
    unusual_text_details = detect_unusual_text_properties(temp_path)

    # Detect redacted content
    redacted_content = detect_redacted_content(temp_path)

    duplicate_text = detect_duplicate_text(temp_path)
 
    hidden_layers = detect_hidden_layers(temp_path)
    
    issues = detect_watermark_tampering(temp_path)

    # Determine overall risk
    overall_risk = determine_overall_risk(
    max(detail["risk"] for detail in metadata_risk_details),
    suspicious_sentences,
    font_details,
    unusual_text_details,
    redacted_content,
    duplicate_text,
    hidden_layers,
    issues
) 
    file_size = os.path.getsize(temp_path)  # File size in bytes
    file_size_mb = round(file_size / (1024 * 1024), 2)  # Convert to MB

    # Two-column layout
    col1, col2 = st.columns([1.2, 2])  # Adjust ratios for spacing

    # Column 1: PDF metadata and analysis
    with col1:
        color, description = generate_risk_band(overall_risk)
        # Add custom CSS for alignment
        st.markdown("""
            <style>
                .details summary {
                    display: flex;
                    align-items: center;
                    font-size: 1.25rem; /* Adjust font size */
                    font-weight: bold;
                    cursor: pointer;
                }
                .details summary::-webkit-details-marker {
                    margin-right: 10px; /* Space between dropdown icon and text */
                    font-size: 1.5rem; /* Adjust icon size */
                }
                .details {
                    margin-top: 7px;
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    background-color: #f9f9f9; /* Light gray background */
                }
                
                
                .details summary:hover {
                    color:rgb(42, 81, 54); /* Change color on hover */
                }
                .indicators summary {
                    display: flex;
                    align-items: center;
                    font-size: 1rem; /* Adjust font size */
                    font-weight: bold;
                    cursor: pointer;
                }
                .indicators summary::-webkit-indicators-marker {
                    margin-right: 8px; /* Space between dropdown icon and text */
                    font-size: 1rem; /* Adjust icon size */
                }
                .indicators {
                    margin-top: 8px;
                    padding: 8px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    background-color: #f9f9f9; /* Light gray background */
                }
                .indicators summary:hover {
                    color:rgb(42, 81, 54); /* Change color on hover */
                }
            </style>
        """, unsafe_allow_html=True)


        # Assuming you have the variables like color, description, metadata, etc., already defined
        st.markdown(f"""
            <div class="left-panel">
                <header class="header">
                    <h1 class="title">Nexyom<span class="ai">.ai</span></h1>
                    <div style="background-color: {color}; padding: 25px; border-radius: 10px; text-align: left; color: black; font-weight: bold;">
                    {description}
                    </div>
                </header>
                <section class="details">
                    <details>
                        <summary><h5>General Information</h5></summary>
                        <p><strong>Author:</strong> {metadata.get('Author', 'Not found')}</p>
                        <p><strong>Title:</strong> {metadata.get('Title', 'Not found')}</p>
                        <p><strong>Producer:</strong> {metadata.get('Producer', 'Not found')}</p>
                        <p><strong>Creation date:</strong> {metadata.get("CreationDate", "NA")}</p>
                        <p><strong>Modified date:</strong> {metadata.get("ModDate", "NA")}</p>
                        <p><strong>Subject:</strong> {metadata.get('Subject', 'Not found')}</p>
                        <p><strong>Page Count:</strong> {len(text_data)} pages</p>
                        <p><strong>File Size:</strong> {round(os.path.getsize(temp_path) / (1024 * 1024), 2)} MB</p>
                    </details>
                </section>
               <section class="indicators">
                    <details>
                        <summary><h2>{len(metadata_risk_details)} Risk Indicators</h2></summary>
                        <ul>
                            {''.join(f'<li>{indicator["reason"]}</li>' for indicator in metadata_risk_details)}
                        </ul>
                    </details>
                </section>
                <section class="indicators">
                    <details>
                        <summary><h2>{len(suspicious_sentences)} Suspicious Sentences</h2></summary>
                        <ul>
                            {''.join(f'<li>**Page {item["page"]}**: {item["sentence"]}<br> **Risk:** {item["risk"]} (Score: {item["score"]:.2f})</li>' for item in suspicious_sentences)}
                        </ul>
                    </details>
                </section>
                <section class="indicators">
                    <details>
                        <summary><h2>{len(font_details)} Fonts and Sizes Detected</h2></summary>
                        <ul>
                            {''.join(f'<li><strong>Page {entry["page"]}</strong> | Font: {entry["font_name"]} | Size: {entry["font_size"]} | Text: {entry["text_snippet"]}<br>**Risk:** {entry["risk"]}</li>' for entry in font_details)}
                        </ul>
                    </details>
                </section>
                <section class="indicators">
                    <details>
                        <summary><h2>{len(issues)} Watermark Tampering Detection Issues</h2></summary>
                        <ul>
                            {"".join(f'<li><strong>{html.escape(issue["message"])} </strong> - <strong>Risk:</strong> {issue["risk"]}</li>' for issue in issues)}
                        </ul>
                    </details>
                </section>
                <section class="indicators">
                    <details>
                        <summary><h2>{len(hidden_layers)} Hidden Layers Detected</h2></summary>
                        <ul>
                            {"".join(f'<li><strong>Page {layer["page"]}</strong> | Element 1: {layer["element_1"]["type"]} at {layer["element_1"]["bbox"]} | '
                                    f'Element 2: {layer["element_2"]["type"]} at {layer["element_2"]["bbox"]}<br>'
                                    f'<strong>Risk:</strong> {layer["risk"]}</li>' for layer in hidden_layers) if len(hidden_layers) > 0 else '<li>No hidden layers detected.</li>'}
                        </ul>
                    </details>
                </section>
                <section class="indicators">
                    <details>
                        <summary><h2>{len(redacted_content)} Redacted Content Detected</h2></summary>
                        <ul>
                            {"".join(f'<li><strong>Page {str(content["page"])}</strong>: Hidden text found under redaction.<br>'
                                    f'<strong>Bounding Box:</strong> {content["bbox"]}<br>'
                                    f'<strong>Hidden Text:</strong> {html.escape(content["hidden_text"])}<br>' 
                                    f'<strong>Risk:</strong> {content["risk"]}</li>' 
                                    for content in redacted_content) if len(redacted_content) > 0 else '<li>No redacted content detected.</li>'}
                        </ul>
                    </details>
                </section>
                <section class="indicators">
                    <details>
                        <summary><h2>{len(unusual_text_details)} Unusual Text Properties Detected</h2></summary>
                        <ul style="list-style-type: none; padding: 0; white-space: nowrap;">
                            {" ".join(f'<li style="display: inline; margin-right: 15px; font-size: smaller; padding-right: 5px;">'
                                    f'<strong>Page {detail.get("page", "Unknown")}</strong>: {detail.get("text_snippet", "No snippet available")} '
                                    f'<strong>Issue:</strong> {detail.get("char_spacing", detail.get("issue", "No issue specified"))} '
                                    + ''.join(f'<strong>{key.replace("_", " ").title()}:</strong> {value} ' for key, value in detail.get("details", {}).items() if value)
                                    + f'<strong>Risk:</strong> {detail.get("risk", "No risk specified")}</li>' 
                                    for detail in unusual_text_details) if len(unusual_text_details) > 0 else '<li>No unusual text properties detected.</li>'}
                        </ul>
                    </details>
                </section>
                <section class="indicators">
                    <details>
                        <summary><h2>{len(duplicate_text)} Duplicate Text/Overlays Detected</h2></summary>
                        <ul>
                            {"".join(f'<li><strong>Page {content["page"]}</strong>: Duplicate text detected.<br>'
                                    f'<strong>Bounding Box:</strong> {content["bbox"]}<br>'
                                    f'<strong>Text 1:</strong> {html.escape(content["text_1"])}<br>'  
                                    f'<strong>Text 2:</strong> {html.escape(content["text_2"])}<br>'  
                                    f'<strong>Risk:</strong> {content["risk"]}</li>' for content in duplicate_text) if len(duplicate_text) > 0 else '<li>No duplicate text or overlays detected.</li>'}
                        </ul>
                    </details>
                </section>
            </div>
        """, unsafe_allow_html=True)
        # Display risk band
        # risk_band_html = generate_risk_band(overall_risk)
        # st.markdown(risk_band_html, unsafe_allow_html=True)

        # # Display metadata analysis with risk indicators
        # st.write("### General Information")
        # for key, value in metadata.items():
        #     st.markdown(f"**{key}:** {value}")
        # for detail in metadata_risk_details:
        #     st.markdown(f"**Risk:** {detail['risk']} - **Reason:** {detail['reason']}")

        # Display suspicious sentences
        # st.write("### Suspicious Sentences")
        # if suspicious_sentences:
        #     for item in suspicious_sentences:
        #         st.write(f"**Page {item['page']}**: {item['sentence']}")
        #         st.markdown(f"**Risk:** {item['risk']} (Score: {item['score']:.2f})")
        # else:
        #     st.success("No suspicious sentences detected.")

    # Column 2: PDF Visualization
    with col2:
        # st.markdown(
        #     """
        #     <style>
        #         .block-container {
        #             background-color: #f7f7f7;  /* Light gray color for PDF viewer */
        #             padding: 20px;
        #             border-radius: 10px;
        #         }
        #     </style>
        #     """, unsafe_allow_html=True
        # )
        # Highlight the PDF
        highlighted_pdf_path = render_highlighted_pdf(temp_path, suspicious_sentences)

        # Display highlighted PDF in PDF Viewer
        pdf_doc = fitz.open(highlighted_pdf_path)
        st.write("### Highlighted PDF Viewer")
        for page_num in range(pdf_doc.page_count):
            page = pdf_doc[page_num]
            pix = page.get_pixmap()
            st.image(pix.tobytes(), use_container_width=True)

        # Provide download link for highlighted PDF
        with open(highlighted_pdf_path, "rb") as file:
            st.download_button(
                label="Download Highlighted PDF",
                data=file,
                file_name="highlighted_output.pdf",
                mime="application/pdf",
            )

    # Cleanup
    os.remove(temp_path)
    pdf_doc.close()
    os.remove(highlighted_pdf_path)
