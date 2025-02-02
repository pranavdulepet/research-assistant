from flask import Flask, request, render_template, jsonify
import os
import fitz  # PyMuPDF
import re
import uuid
import math
import subprocess

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/figures'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Threshold (in PDF points) above which we consider a figure "special" and convert it to SVG.
SPECIAL_THRESHOLD = 300  

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

    # Save uploaded PDF to a temporary location
    temp_pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_upload.pdf')
    pdf_file.save(temp_pdf_path)

    extracted_data = extract_text_and_figures(temp_pdf_path)
    # Optionally remove the temporary PDF: os.remove(temp_pdf_path)
    return jsonify(extracted_data)

def convert_region_to_svg(pdf_path, page_number, bbox, output_svg):
    """
    Convert a region of the PDF page to an SVG file using pdftocairo.
    bbox is a tuple (x0, y0, x1, y1); coordinates are rounded to integers.
    """
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
    """
    Optionally correct common mis-extraction issues.
    For example, replace caret (ˆ) sequences with a literal caret.
    (You can adjust this function as needed.)
    """
    # In this example, we'll simply replace the caret (ˆ) with '^'
    text = text.replace("ˆ", "^")
    # And replace "D X" with "DX" if that's desired.
    text = text.replace("D X", "DX")
    return text

def extract_text_and_figures(pdf_path):
    doc = fitz.open(pdf_path)
    result = {"pages": []}

    # Regex for LaTeX-style figure references (e.g., "Figure 1", "Fig. 2.1", etc.)
    figure_ref_regex = re.compile(
        r'(?:Figure|Fig\.?|FIGURE)\s+(\d+(\.\d+)*[a-z]?)',
        re.IGNORECASE
    )

    for page_index, page in enumerate(doc):
        # Get text blocks with bounding boxes
        text_blocks = page.get_text("blocks")
        # Get image info; each info's bbox is a tuple (x0, y0, x1, y1)
        image_info_list = page.get_image_info(xrefs=True)

        processed_blocks = []
        # Process each text block as plaintext.
        for block in text_blocks:
            block_text = block[4]
            # Optionally fix extraction issues:
            block_text = fix_math_extraction(block_text)
            processed_blocks.append(block_text)

        # Find figure references in the original text (for positioning)
        references_on_page = []
        for block in text_blocks:
            x0, y0, x1, y1, block_text, block_id = block[0], block[1], block[2], block[3], block[4], block[5]
            for match in figure_ref_regex.finditer(block_text):
                ref_str = match.group(0)  # e.g. "Figure 2.1"
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

            # For large figures, convert to SVG; otherwise extract as a raster image.
            if width >= SPECIAL_THRESHOLD or height >= SPECIAL_THRESHOLD:
                filename = f"{uuid.uuid4()}.svg"
                output_svg_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                convert_region_to_svg(pdf_path, page_index + 1, (bbox.x0, bbox.y0, bbox.x1, bbox.y1), output_svg_path)
                figure_url = f"/static/figures/{filename}"
            else:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                filename = f"{uuid.uuid4()}.{image_ext}"
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                figure_url = f"/static/figures/{filename}"

            figures_data.append({
                "url": figure_url,
                "bbox": (bbox.x0, bbox.y0, bbox.x1, bbox.y1),
                "center": bbox_center,
                "assigned_ref": None
            })

        # For each figure reference, match it to the nearest unassigned image.
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

        # Build mapping: reference text → figure URL.
        ref_to_url_map = { img["assigned_ref"]: img["url"]
                           for img in figures_data if img["assigned_ref"] }

        # Reconstruct the page HTML from the processed blocks,
        # replacing figure references with clickable links.
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

if __name__ == '__main__':
    app.run(debug=True)
