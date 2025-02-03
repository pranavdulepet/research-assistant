from flask import Flask, request, render_template, jsonify
import os
import fitz  # PyMuPDF
import re
import uuid
import math
import subprocess
import json
from dotenv import load_dotenv
import google.generativeai as genai

app = Flask(__name__)

# Directories for uploads and extracted figures.
app.config['UPLOAD_FOLDER'] = 'static/figures'
app.config['UPLOAD_PDF'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['UPLOAD_PDF'], exist_ok=True)

# File where conversation data will be stored.
CONVERSATIONS_FILE = "conversations.json"

SPECIAL_THRESHOLD = 300  # threshold (in PDF points) for converting a figure to SVG

load_dotenv()

# Configure Google Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-pro")

def load_conversations():
    if os.path.exists(CONVERSATIONS_FILE):
        with open(CONVERSATIONS_FILE, "r") as f:
            return json.load(f)
    return []

def save_conversations(conversations):
    with open(CONVERSATIONS_FILE, "w") as f:
        json.dump(conversations, f, indent=2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'pdfFile' not in request.files:
        return "No file part in the request", 400
    pdf_file = request.files['pdfFile']
    if pdf_file.filename == '':
        return "No selected file", 400

    # Save the PDF with a unique filename.
    unique_pdf_name = f"{uuid.uuid4()}.pdf"
    pdf_path = os.path.join(app.config['UPLOAD_PDF'], unique_pdf_name)
    pdf_file.save(pdf_path)
    
    # Run extraction on the PDF.
    extraction_data = extract_text_and_figures(pdf_path)
    
    # Return the PDF URL and extraction data.
    result = {
        "pdf_url": f"/{pdf_path}",
        "extraction": extraction_data
    }
    return jsonify(result)

def convert_region_to_svg(pdf_path, page_number, bbox, output_svg):
    # Round coordinates to integers.
    x0, y0, x1, y1 = bbox
    x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    width = x1 - x0
    height = y1 - y0

    command = [
        "pdftocairo",
        "-svg",
        "-f", str(page_number),
        "-l", str(page_number),
        "-x", str(x0),
        "-y", str(y0),
        "-W", str(width),
        "-H", str(height),
        pdf_path,
        output_svg
    ]
    subprocess.run(command, check=True)

def fix_math_extraction(text):
    # Simple fixes: replace caret characters and "D X" if needed.
    text = text.replace("ˆ", "^")
    text = text.replace("D X", "DX")
    return text

def extract_text_and_figures(pdf_path):
    doc = fitz.open(pdf_path)
    result = {"pages": []}
    figure_ref_regex = re.compile(
        r'(?:Figure|Fig\.?|FIGURE)\s+(\d+(\.\d+)*[a-z]?)',
        re.IGNORECASE
    )

    for page_index, page in enumerate(doc):
        text_blocks = page.get_text("blocks")
        image_info_list = page.get_image_info(xrefs=True)
        processed_blocks = []
        for block in text_blocks:
            block_text = block[4]
            block_text = fix_math_extraction(block_text)
            processed_blocks.append(block_text)

        references_on_page = []
        for block in text_blocks:
            x0, y0, x1, y1, block_text, block_id = block[0], block[1], block[2], block[3], block[4], block[5]
            for match in figure_ref_regex.finditer(block_text):
                ref_str = match.group(0)
                ref_center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
                references_on_page.append({
                    "reference_text": ref_str,
                    "bbox": (x0, y0, x1, y1),
                    "center": ref_center
                })

        figures_data = []
        for info in image_info_list:
            xref = info["xref"]
            bbox_tuple = info["bbox"]
            bbox = fitz.Rect(bbox_tuple)
            bbox_center = ((bbox.x0 + bbox.x1) / 2.0, (bbox.y0 + bbox.y1) / 2.0)
            width = bbox.width
            height = bbox.height

            if width >= SPECIAL_THRESHOLD or height >= SPECIAL_THRESHOLD:
                filename = f"{uuid.uuid4()}.svg"
                output_svg_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                convert_region_to_svg(pdf_path, page_index + 1, (bbox.x0, bbox.y0, bbox.x1, bbox.y1), output_svg_path)
                figure_url = f"/{os.path.join(app.config['UPLOAD_FOLDER'], filename)}"
            else:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                filename = f"{uuid.uuid4()}.{image_ext}"
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                figure_url = f"/{image_path}"

            figures_data.append({
                "url": figure_url,
                "bbox": (bbox.x0, bbox.y0, bbox.x1, bbox.y1),
                "center": bbox_center,
                "assigned_ref": None
            })

        for ref_obj in references_on_page:
            ref_center = ref_obj["center"]
            best_image = None
            best_dist = float('inf')
            for img_obj in figures_data:
                if img_obj["assigned_ref"] is not None:
                    continue
                dx = ref_center[0] - img_obj["center"][0]
                dy = ref_center[1] - img_obj["center"][1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < best_dist:
                    best_dist = dist
                    best_image = img_obj
            if best_image:
                best_image["assigned_ref"] = ref_obj["reference_text"]

        ref_to_url_map = { img["assigned_ref"]: img["url"]
                           for img in figures_data if img["assigned_ref"] }

        replaced_page_html = ""
        for block_text in processed_blocks:
            def replace_ref(match):
                ref_str = match.group(0)
                url = ref_to_url_map.get(ref_str)
                if url:
                    return f"<a href='#' class='figure-link' data-figure-url='{url}'>{ref_str}</a>"
                return ref_str
            block_html = figure_ref_regex.sub(replace_ref, block_text)
            replaced_page_html += block_html + "<br><br>"

        result["pages"].append({
            "page_number": page_index + 1,
            "text": replaced_page_html,
            "figures": [{"ref": img["assigned_ref"], "url": img["url"]} for img in figures_data]
        })

    doc.close()
    return result

@app.route("/conversations", methods=["GET"])
def get_conversations():
    convs = load_conversations()
    return jsonify(convs)

@app.route("/conversations", methods=["POST"])
def save_conversations_endpoint():
    convs = request.get_json()
    save_conversations(convs)
    return jsonify({"status": "ok"})

@app.route("/delete-static-files", methods=["POST"])
def delete_static_files():
    pdf_url = request.json.get('pdf_url')
    figure_urls = request.json.get('figure_urls', [])
    
    # Delete PDF file
    if pdf_url:
        pdf_path = pdf_url.lstrip('/')
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
    
    # Delete figure files
    for figure_url in figure_urls:
        figure_path = figure_url.lstrip('/')
        if os.path.exists(figure_path):
            os.remove(figure_path)
    
    return jsonify({"status": "ok"})

@app.route("/verify-annotation", methods=["POST"])
def verify_annotation():
    print("\n=== NEW VERIFICATION REQUEST RECEIVED ===")
    
    data = request.get_json()
    print("Request data received:", bool(data))
    
    annotation_text = data.get("selectedText", "")
    comment = data.get("comment", "")
    context = data.get("context", "")
    figures = data.get("figures", [])
    
    print(f"Processing request for annotation: '{annotation_text[:50]}...'")
    print(f"Number of figures in context: {len(figures)}")
    
    # Build context with figures
    figures_context = "\n".join([f"Figure {fig['ref']}: Located at page {fig['page_number']}" 
                                for fig in figures if fig.get('ref')])
    
    prompt = f"""
    As an AI research assistant, please verify the following annotation in the context of this research paper:
    
    Selected Text: "{annotation_text}"
    User's Comment/Claim: "{comment}"
    
    Context from the paper:
    {context}
    
    Available Figures in Context:
    {figures_context}
    
    Please:
    1. Verify if the comment/claim is accurate based on the selected text, context, and available figures
    2. Provide evidence supporting or contradicting the claim
    3. Suggest any corrections if needed
    
    Format your response in clear sections.
    """
    
    print("\n=== SENDING REQUEST TO GEMINI ===")
    try:
        print("Waiting for Gemini response...")
        response = model.generate_content(prompt)
        print("Response received from Gemini!")
        print("\n=== AI RESPONSE ===")
        print(response.text)
        print("\n=== END OF RESPONSE ===")
        
        return jsonify({
            "verified": True,
            "explanation": response.text
        })
    except Exception as e:
        print("\n=== ERROR OCCURRED ===")
        print("Error type:", type(e).__name__)
        print("Error message:", str(e))
        print("=== END OF ERROR ===\n")
        return jsonify({
            "verified": False,
            "error": str(e)
        }), 500

@app.route("/answer-question", methods=["POST"])
def answer_question():
    print("\n=== NEW ANSWER REQUEST RECEIVED ===")
    
    data = request.get_json()
    print("Request data received:", bool(data))
    
    question = data.get("question", "")
    context = data.get("context", "")
    selected_text = data.get("selectedText", "")
    figures = data.get("figures", [])
    
    print(f"Processing answer for question: '{question}'")
    print(f"Number of figures in context: {len(figures)}")
    
    # Build context with figures
    figures_context = "\n".join([f"Figure {fig['ref']}: Located at page {fig['page_number']}" 
                               for fig in figures if fig.get('ref')])
    
    prompt = f"""
    As an AI research assistant, please answer the following question about this research paper:
    
    Question: "{question}"
    
    Selected Text: "{selected_text}"
    
    Context from the paper:
    {context}
    
    Available Figures in Context:
    {figures_context}
    
    Please provide a clear and detailed answer based on the provided context and figures.
    """
    
    try:
        print("Waiting for Gemini response...")
        response = model.generate_content(prompt)
        print("Response received from Gemini!")
        
        return jsonify({
            "answer": response.text
        })
    except Exception as e:
        print("\n=== ERROR OCCURRED ===")
        print("Error type:", type(e).__name__)
        print("Error message:", str(e))
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == '__main__':
    print("Starting Flask server in debug mode...")
    app.run(debug=True, port=5000)
