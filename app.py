from flask import Flask, request, render_template, jsonify, Response, stream_with_context
import os
import fitz 
import re
import uuid
import math
import subprocess
import json
from dotenv import load_dotenv
from google import genai
import shutil
from pathlib import Path
from datetime import datetime
import openai
import httpx
from openai import OpenAI
from duckduckgo_search import DDGS
import logging
import sqlite3
from bs4 import BeautifulSoup
import threading
from queue import Queue
import time
from combined_extraction import extract_references_from_pdf, get_open_access_pdf_url, fetch_paper_metadata
import numpy as np
from typing import List, Dict
import asyncio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
from werkzeug.utils import secure_filename

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
)

app = Flask(__name__)

# Canonical storage root for all uploads/artifacts.
app.config['STORAGE_BASE'] = 'static/uploads'
os.makedirs(app.config['STORAGE_BASE'], exist_ok=True)

# File where conversation data will be stored.
CONVERSATIONS_FILE = "conversations.json"

SPECIAL_THRESHOLD = 300  # threshold (in PDF points) for converting a figure to SVG

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Warn if legacy duplicate upload dir exists (should not be used)
try:
    legacy_uploads = Path(__file__).resolve().parent / "app" / "static" / "uploads"
    if legacy_uploads.exists():
        logging.warning("Legacy uploads directory exists at %s (canonical is %s)", legacy_uploads, app.config['STORAGE_BASE'])
except Exception:
    pass

load_dotenv()
def run_async(coro):
    """
    Run an async coroutine from sync Flask handlers safely.
    Avoids asyncio.run() pitfalls if an event loop is already present.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

@app.after_request
def add_security_headers(resp):
    # Basic hardening headers (tweak if you embed in iframes etc.)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    resp.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    resp.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
    return resp

@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "File too large"}), 413

@app.errorhandler(404)
def not_found(_e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(e):
    logging.exception("Unhandled server error: %s", str(e))
    return jsonify({"error": "Internal server error"}), 500


# After load_dotenv(), add these debug prints:
logging.info("Environment Variables Check:")
logging.info("GEMINI_API_KEY present: %s", bool(os.getenv('GEMINI_API_KEY')))
logging.info("OPENAI_API_KEY present: %s", bool(os.getenv('OPENAI_API_KEY')))

# Configure AI Providers
# "enabled" here means "supported by the app" (not "server has a key").
# This app supports BYOK (Bring Your Own Key): users can provide their own key per request.
AI_PROVIDERS = {
    'gemini': {
        'id': 'gemini',
        'name': 'Google Gemini',
        'enabled': True,
        'api_key': os.getenv("GEMINI_API_KEY")  # optional server key
    },
    'openai': {
        'id': 'openai',
        'name': 'OpenAI',
        'enabled': True,
        'api_key': os.getenv("OPENAI_API_KEY")  # optional server key
    }
}

# Initialize server-side clients only if a server API key is present.
# For BYOK, we create a short-lived client per request (no key storage).
gemini_client = None
openai_client = None
if AI_PROVIDERS.get('gemini', {}).get('api_key'):
    # Using new google-genai SDK
    gemini_client = genai.Client(api_key=AI_PROVIDERS['gemini']['api_key'])
    print("Gemini client initialized successfully with server key (google-genai SDK)")

if AI_PROVIDERS.get('openai', {}).get('api_key'):
    openai_client = OpenAI(api_key=AI_PROVIDERS['openai']['api_key'])
    print("OpenAI client initialized successfully with server key.")

def _extract_api_key(provider: str, data: dict | None = None) -> str | None:
    """
    Extract a user-provided API key from request JSON or headers.
    Never store it server-side; only use for the duration of a single request.
    """
    key = None
    if isinstance(data, dict):
        key = (
            data.get("apiKey")
            or data.get("api_key")
            or data.get(f"{provider}_api_key")
            or data.get(f"{provider}ApiKey")
        )
    # Provider-specific header names (nice for non-browser clients)
    hdr = (
        request.headers.get("X-AI-API-Key")
        or request.headers.get("X-Api-Key")
        or request.headers.get(f"X-{provider.upper()}-API-Key")
        or request.headers.get(f"X-{provider}-api-key")
    )
    key = (hdr or key or "")
    key = key.strip()
    return key or None

def _effective_api_key(provider: str, data: dict | None = None) -> str | None:
    """User key wins; otherwise fall back to server env key if configured."""
    return _extract_api_key(provider, data) or AI_PROVIDERS.get(provider, {}).get("api_key")

def _ensure_provider_has_key(provider: str, data: dict | None = None):
    if not _effective_api_key(provider, data):
        raise ValueError(
            f"No API key provided for {AI_PROVIDERS.get(provider, {}).get('name', provider)}. "
            f"Paste your key in the UI (BYOK), or configure a server key via environment variables."
        )

def load_conversations():
    if os.path.exists(CONVERSATIONS_FILE):
        with open(CONVERSATIONS_FILE, "r") as f:
            return json.load(f)
    return []

def save_conversations(conversations):
    with open(CONVERSATIONS_FILE, "w") as f:
        json.dump(conversations, f, indent=2)

def _safe_filename(name: str) -> str:
    name = (name or "").strip() or "export"
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name[:120]

@app.route("/export/<upload_id>", methods=["GET"])
def export_upload(upload_id):
    fmt = (request.args.get("format") or "md").lower()
    convs = load_conversations()
    conv = next((c for c in convs if c.get("id") == upload_id), None)
    if not conv:
        return jsonify({"error": "Upload not found"}), 404

    title = conv.get("title") or upload_id
    annotations = conv.get("annotations") or []

    if fmt == "json":
        payload = {
            "id": conv.get("id"),
            "title": title,
            "pdf_url": conv.get("pdf_url"),
            "exported_at": datetime.utcnow().isoformat() + "Z",
            "annotations": annotations,
        }
        body = json.dumps(payload, indent=2)
        filename = _safe_filename(title) + ".annotations.json"
        return Response(
            body,
            mimetype="application/json",
            headers={"Content-Disposition": f'attachment; filename=\"{filename}\"'},
        )

    # Markdown default
    by_page = {}
    for a in annotations:
        page = a.get("page")
        by_page.setdefault(page, []).append(a)

    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- Exported: {datetime.utcnow().isoformat()}Z")
    if conv.get("pdf_url"):
        lines.append(f"- PDF: {conv.get('pdf_url')}")
    lines.append("")

    for page in sorted([p for p in by_page.keys() if p is not None]):
        lines.append(f"## Page {page}")
        lines.append("")
        for a in by_page[page]:
            a_type = a.get("type", "note")
            sel = a.get("selectedText", "").strip()
            comment = a.get("comment", "").strip()
            lines.append(f"### {a_type.title()}")
            if sel:
                lines.append(f"> {sel}")
                lines.append("")
            if comment:
                lines.append(comment)
                lines.append("")
            if a_type == "note" and a.get("aiVerification"):
                lines.append("**AI Verification:**")
                lines.append("")
                lines.append(a.get("aiVerification"))
                lines.append("")
            if a_type == "question" and a.get("aiAnswer"):
                lines.append("**AI Answer:**")
                lines.append("")
                lines.append(a.get("aiAnswer"))
                lines.append("")

    body = "\n".join(lines).strip() + "\n"
    filename = _safe_filename(title) + ".annotations.md"
    return Response(
        body,
        mimetype="text/markdown",
        headers={"Content-Disposition": f'attachment; filename=\"{filename}\"'},
    )

@app.route('/')
def index():
    return render_template('index.html')

# Add these global variables at the top level
background_tasks = {}
background_tasks_lock = threading.Lock()
progress_updates = Queue()

# Email for Unpaywall API (configure via environment variable)
UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL", "your-email@example.com")

# Add this new class to manage background tasks
class BackgroundTask:
    def __init__(self, upload_id):
        self.upload_id = upload_id
        self.progress = 0
        self.status = "Processing"
        self.result = None
        self.error = None

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'pdfFile' not in request.files:
        logging.error("No file part in the request")
        return "No file part in the request", 400
    
    pdf_file = request.files['pdfFile']
    if pdf_file.filename == '':
        logging.error("No selected file")
        return "No selected file", 400

    # Basic validation
    filename = secure_filename(pdf_file.filename or "")
    if not filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF uploads are supported"}), 400
    if pdf_file.mimetype and pdf_file.mimetype not in ("application/pdf", "application/octet-stream"):
        return jsonify({"error": f"Invalid content type: {pdf_file.mimetype}"}), 400
    
    # Create unique ID and directory structure
    upload_id = str(uuid.uuid4())
    paths = create_upload_structure(upload_id)
    logging.info("Created upload structure for ID: %s", upload_id)
    
    # Save the PDF
    pdf_path = paths['pdf_path'] / 'original.pdf'
    pdf_file.save(pdf_path)
    logging.info("Saved PDF to %s", pdf_path)
    
    # Extract data (text and figures)
    extraction_data = extract_text_and_figures(str(pdf_path), upload_id, paths['figures_path'])
    logging.info("Completed text and figure extraction")
    
    # Initialize background task (thread-safe)
    task = BackgroundTask(upload_id)
    with background_tasks_lock:
        background_tasks[upload_id] = task
    
    # Start background processing
    thread = threading.Thread(
        target=process_references_background,
        args=(str(pdf_path), upload_id, paths['citations_path'], task)
    )
    thread.start()
    
    result = {
        "upload_id": upload_id,
        "pdf_url": f"/static/uploads/{upload_id}/pdf/original.pdf",
        "extraction": extraction_data,
        "task_id": upload_id  # Return task_id for progress tracking
    }
    
    logging.info("Upload endpoint completed for ID: %s", upload_id)
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

def extract_text_and_figures(pdf_path, upload_id, figures_path):
    doc = fitz.open(pdf_path)
    result = {"pages": []}
    figure_ref_regex = re.compile(
        r'(?:Figure|Fig\.?|FIGURE)\s+(\d+(\.\d+)*[a-z]?)',
        re.IGNORECASE
    )
    figure_caption_regex = re.compile(
        r'^\s*(?:Figure|Fig\.?|FIGURE)\s+(\d+(\.\d+)*[a-z]?)\s*[:\.\-]\s*(.+)$',
        re.IGNORECASE
    )

    def _figure_num_key(s: str):
        if not s:
            return None
        m = re.search(r'(\d+(\.\d+)*[a-z]?)', s, re.IGNORECASE)
        return m.group(1).lower() if m else None

    for page_index, page in enumerate(doc):
        text_blocks = page.get_text("blocks")
        image_info_list = page.get_image_info(xrefs=True)
        processed_blocks = []
        for block in text_blocks:
            block_text = block[4]
            block_text = fix_math_extraction(block_text)
            processed_blocks.append(block_text)

        # Extract likely figure captions on this page
        captions_by_key = {}
        for t in processed_blocks:
            m = figure_caption_regex.match((t or "").strip())
            if m:
                key = (m.group(1) or "").lower()
                if key and key not in captions_by_key:
                    captions_by_key[key] = (t or "").strip()

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
            width = bbox.width
            height = bbox.height
            bbox_center = ((bbox.x0 + bbox.x1) / 2.0, (bbox.y0 + bbox.y1) / 2.0)

            try:
                if width >= SPECIAL_THRESHOLD or height >= SPECIAL_THRESHOLD:
                    filename = f"{uuid.uuid4()}.svg"
                    output_svg_path = figures_path / filename
                    convert_region_to_svg(pdf_path, page_index + 1, (bbox.x0, bbox.y0, bbox.x1, bbox.y1), str(output_svg_path))
                    figure_url = f"/static/uploads/{upload_id}/figures/{filename}"
                else:
                    base_image = doc.extract_image(xref)
                    if isinstance(base_image, bool):
                        # Skip this image if extraction failed
                        logging.warning(f"Failed to extract image with xref {xref} on page {page_index + 1}")
                        continue
                        
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    filename = f"{uuid.uuid4()}.{image_ext}"
                    image_path = figures_path / filename
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                    figure_url = f"/static/uploads/{upload_id}/figures/{filename}"

                figures_data.append({
                    "url": figure_url,
                    "bbox": (bbox.x0, bbox.y0, bbox.x1, bbox.y1),
                    "center": bbox_center,
                    "assigned_ref": None
                })
            except Exception as e:
                logging.error(f"Error processing image on page {page_index + 1}: {str(e)}")
                continue

        for ref_obj in references_on_page:
            ref_center = ref_obj["center"]
            best_image = None
            best_dist = float('inf')
            for img_obj in figures_data:
                if img_obj["assigned_ref"]:
                    continue
                dx = ref_center[0] - img_obj["center"][0]
                dy = ref_center[1] - img_obj["center"][1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < best_dist:
                    best_dist = dist
                    best_image = img_obj
            if best_image:
                best_image["assigned_ref"] = ref_obj["reference_text"]

        # Attach captions (if found) to assigned figures
        for img_obj in figures_data:
            ref = img_obj.get("assigned_ref")
            key = _figure_num_key(ref)
            img_obj["caption"] = captions_by_key.get(key) if key else None

        ref_to_url_map = {img["assigned_ref"]: img["url"]
                         for img in figures_data if img["assigned_ref"]}

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
            "figures": [{"ref": img["assigned_ref"], "url": img["url"], "caption": img.get("caption")} for img in figures_data]
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

@app.route("/delete-upload/<upload_id>", methods=["POST"])
def delete_upload(upload_id):
    try:
        # Get the base path for this upload
        base_path = Path(app.config['STORAGE_BASE']) / upload_id
        
        if base_path.exists():
            # Remove the entire directory structure for this upload including citations, chats, pdf, and figures.
            shutil.rmtree(base_path)
            logging.info("Deleted folder for upload_id: %s", upload_id)
            
            # Delete corresponding cited papers from the database
            delete_cited_papers(upload_id)
            
            # Clean up document store for this upload
            with document_stores_lock:
                if upload_id in document_stores:
                    del document_stores[upload_id]
                    logging.info("Cleaned up document store for upload_id: %s", upload_id)
            
            # Clean up background task for this upload
            with background_tasks_lock:
                if upload_id in background_tasks:
                    del background_tasks[upload_id]
                    logging.info("Cleaned up background task for upload_id: %s", upload_id)
            
            # Update conversations.json accordingly
            conversations = load_conversations()
            conversations = [c for c in conversations if c['id'] != upload_id]
            save_conversations(conversations)
            
            return jsonify({"status": "ok"})
        else:
            return jsonify({"error": "Upload not found"}), 404
    except Exception as e:
        logging.exception("Error occurred while processing delete_upload")
        return jsonify({"error": str(e)}), 500

@app.route("/verify-annotation", methods=["POST"])
def verify_annotation():
    data = request.get_json() or {}
    provider = data.get("provider")
    upload_id = data.get("uploadId")
    
    if not upload_id:
        return jsonify({"error": "Upload ID is required"}), 400
    
    if not provider or provider not in AI_PROVIDERS:
        return jsonify({"error": "Invalid AI provider"}), 400

    try:
        _ensure_provider_has_key(provider, data)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
    annotation_text = data.get("selectedText", "")
    comment = data.get("comment", "")
    context = data.get("context", "")
    figures = data.get("figures", [])
    
    # Perform web search for relevant information
    try:
        web_results = run_async(perform_web_search(f"{comment} {annotation_text}"))
        web_context = "\n\nRelevant information from web search:\n" + web_results if web_results else ""
    except Exception as e:
        print(f"Web search error: {str(e)}")
        web_context = ""
    
    # Build context with figures
    figures_context = "\n".join([
        f"Figure {fig['ref']} (page {fig.get('page_number')}): {fig.get('caption') or 'No caption found.'}"
        for fig in figures if fig.get('ref')
    ])
    
    # Get referenced papers context
    referenced_papers_context = get_referenced_papers_context(upload_id)
    
    prompt = f"""
    You are an expert AI research assistant. Your task is to verify the accuracy of a user's comment or claim regarding a specific annotation in a paper. Use only the provided sources—selected text, context from the paper, referenced papers, available figures, and web search results—along with any relevant general knowledge.

    **Inputs Provided:**
    - **Selected Text:** "{annotation_text}"
    - **User's Comment/Claim:** "{comment}"
    - **Context from the Paper:**  
    {context}
    - **Available Figures in Context:**  
    {figures_context}
    - **Referenced Papers Context:**  
    {referenced_papers_context}
    - **Web Search Results:**  
    {web_context}

    **Instructions:**
    1. **Verification:**  
    Verify whether the user's comment/claim about the selected text is accurate. Provide a highly detailed and specific analysis.

    2. **Evidence from the Paper:**  
    Quote and cite specific parts of the paper that support your verification.

    3. **Evidence from Referenced Papers:**  
    Quote and cite specific parts from referenced papers that support your verification.

    4. **Evidence from Web Sources:**  
    Clearly cite any web sources used, including URLs when available. Use the format: [Source Title](URL).

    5. **General Knowledge:**  
    Indicate any additional relevant information from your training.

    6. **Suggested Corrections:**  
    If applicable, provide any corrections or clarifications to the user's comment/claim.

    **Formatting Requirements:**
    - Structure your response using Markdown for better readability.
    - When citing web sources, use the format: [Source Title](URL).
    - When citing referenced papers, use the format: [Paper Title] (Year).

    **Important:**
    - Base your analysis solely on the provided inputs.
    - If the available information is insufficient to verify the claim, explicitly state that.
    - Do not hallucinate or include any unverified details.
    """

    
    print("\n=== SENDING REQUEST TO GEMINI ===")
    try:
        print("Waiting for Gemini response...")
        api_key = _effective_api_key(provider, data)
        response = run_async(get_ai_response(provider, prompt, api_key=api_key))
        print("Response received from Gemini!")
        print("\n=== AI RESPONSE ===")
        print(response)
        print("\n=== END OF RESPONSE ===")
        
        return jsonify({
            "verified": True,
            "explanation": response
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

# Add this new class for document chunking and embedding
class DocumentStore:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.chunks: List[str] = []
        self.vectors = None
        
    def add_document(self, text: str, chunk_size: int = 512, overlap: int = 128):
        """Split document into overlapping chunks and compute TF-IDF vectors."""
        # Split into sentences first
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            current_chunk.append(sentence)
            current_length += len(sentence)
            
            if current_length >= chunk_size:
                chunk_text = '. '.join(current_chunk)
                chunks.append(chunk_text)
                # Keep last few sentences for overlap
                current_chunk = current_chunk[-overlap:]
                current_length = sum(len(s) for s in current_chunk)
        
        # Add final chunk if it exists
        if current_chunk:
            chunks.append('. '.join(current_chunk))
        
        # Store chunks and compute TF-IDF vectors
        self.chunks = chunks
        self.vectors = self.vectorizer.fit_transform(chunks)
        
    def search(self, query: str, k: int = 3) -> List[str]:
        """Search for most relevant chunks given a query."""
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.vectors)
        top_k_indices = similarities[0].argsort()[-k:][::-1]
        return [self.chunks[i] for i in top_k_indices]

# Add this to store document embeddings
document_stores = {}
document_stores_lock = threading.Lock()

# Separate stores for whole-paper chat retrieval (avoid mixing with per-request context stores)
paper_document_stores = {}
paper_document_stores_lock = threading.Lock()

def _html_to_text(html: str) -> str:
    """Convert extracted HTML-ish page text to plaintext for retrieval."""
    try:
        soup = BeautifulSoup(html or "", "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception:
        # Fallback: strip tags crudely
        return re.sub(r"<[^>]+>", " ", html or "")

def _paper_context_to_plaintext(paper_context: Dict) -> str:
    pages = paper_context.get("pages", []) if isinstance(paper_context, dict) else []
    texts = []
    for p in pages:
        if isinstance(p, dict):
            texts.append(_html_to_text(p.get("text", "")))
    return "\n".join(t for t in texts if t)

def _text_digest(text: str, max_chars: int = 200_000) -> str:
    """Stable digest for cache invalidation; hashes a prefix for speed."""
    t = (text or "")[:max_chars]
    return hashlib.sha256(t.encode("utf-8", errors="ignore")).hexdigest()

# Update the answer_question function
@app.route("/answer-question", methods=["POST"])
def answer_question():
    try:
        data = request.get_json() or {}
        upload_id = data.get("uploadId")
        
        if not upload_id:
            return jsonify({"error": "Upload ID is required"}), 400
            
        provider = data.get("provider")
        if not provider or provider not in AI_PROVIDERS:
            return jsonify({"error": "Invalid AI provider"}), 400
        try:
            _ensure_provider_has_key(provider, data)
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        
        question = data.get("question", "")
        context = data.get("context", "")
        selected_text = data.get("selectedText", "")
        figures = data.get("figures", [])
        
        # Initialize or get document store for this upload (thread-safe)
        with document_stores_lock:
            if upload_id not in document_stores:
                doc_store = DocumentStore()
                doc_store.add_document(context)
                document_stores[upload_id] = doc_store
            
            # Get most relevant chunks for the question
            relevant_chunks = document_stores[upload_id].search(question, k=3)
        context_text = "\n\n".join(relevant_chunks)
        
        # Build context with figures
        figures_context = "\n".join([
            f"Figure {fig['ref']} (page {fig.get('page_number')}): {fig.get('caption') or 'No caption found.'}"
            for fig in figures if fig.get('ref')
        ])
        
        # Get referenced papers context
        referenced_papers_context = get_referenced_papers_context(upload_id)
        
        print(f"Question: {question}\n\n")
        print(f"Selected Text: {selected_text}\n\n")
        print(f"Context: {context_text}\n\n")
        print(f"Figures Context: {figures_context}\n\n")
        print(f"Referenced Papers Context: {referenced_papers_context}\n\n")

        prompt = f"""
        You are an expert AI research assistant tasked with answering questions about a specific annotated part of a paper. Use the following information to craft your response:

        **Inputs Provided:**
        - **Question:** "{question}"
        - **Selected Text:** "{selected_text}"
        - **Context from the Paper:**  
        {context_text}
        - **Available Figures in Context:**  
        {figures_context}
        - **Referenced Papers Context:**  
        {referenced_papers_context}

        **Your Response Must Include the Following Sections:**

        1. **Answer:**  
        Provide a clear, detailed, and highly specific answer to the question.

        2. **Evidence from the Paper:**  
        Quote and cite specific parts of the paper that support your answer.

        3. **Evidence from Referenced Papers:**  
        Quote and cite specific parts from referenced papers that support your answer.

        4. **General Knowledge:**  
        Indicate any additional relevant information from your training data.

        5. **Additional Context:**  
        Include any relevant figures or sections that might be helpful for understanding.

        **Formatting Requirements:**
        - Structure your response using Markdown for better readability.
        - When citing web sources, use the format: [Source Title](URL).
        - When citing referenced papers, use the format: [Paper Title] (Year).

        **Important:**  
        - Base your response solely on the provided inputs and your general knowledge.
        - If the available information is insufficient, clearly state that you cannot fully answer the question.
        - Avoid hallucinating or making up details.

        """
        
        try:
            api_key = _effective_api_key(provider, data)
            response = run_async(get_ai_response(provider, prompt, api_key=api_key))
            return jsonify({"answer": response})
        except Exception as e:
            print(f"Error getting AI response: {str(e)}")
            return jsonify({"error": str(e)}), 500
            
    except Exception as e:
        print(f"Error in answer_question route: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@app.route("/chat-with-paper", methods=["POST"])
def chat_with_paper():
    data = request.get_json() or {}
    upload_id = data.get("uploadId")
    
    if not upload_id:
        return jsonify({"error": "Upload ID is required"}), 400
    
    provider = data.get("provider")
    if not provider or provider not in AI_PROVIDERS:
        return jsonify({"error": "Invalid AI provider"}), 400
    try:
        _ensure_provider_has_key(provider, data)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    message = data.get("message", "")
    paper_context = data.get("paper_context", {})
    annotations_context = data.get("annotations_context", "")
    
    # Build plaintext and retrieve only the most relevant chunks (avoid sending full paper each request)
    full_text_plain = _paper_context_to_plaintext(paper_context)
    digest = _text_digest(full_text_plain)
    with paper_document_stores_lock:
        cached = paper_document_stores.get(upload_id)
        if not cached or cached.get("digest") != digest:
            ds = DocumentStore()
            # Larger chunks for chat; overlap via the existing sentence-window behavior
            ds.add_document(full_text_plain, chunk_size=1200, overlap=4)
            paper_document_stores[upload_id] = {"digest": digest, "store": ds}
        doc_store = paper_document_stores[upload_id]["store"]

    try:
        relevant_chunks = doc_store.search(message, k=5) if full_text_plain else []
    except Exception:
        relevant_chunks = []

    context_text = "\n\n---\n\n".join(relevant_chunks) if relevant_chunks else (full_text_plain[:6000] if full_text_plain else "")
    
    # Get referenced papers context
    referenced_papers_context = get_referenced_papers_context(upload_id)
    
    # Perform web search
    try:
        web_results = run_async(perform_web_search(message))
        web_context = "\n\nRelevant information from web search:\n" + web_results if web_results else ""
    except Exception as e:
        print(f"Web search error: {str(e)}")
        web_context = ""
    
    prompt = f"""
    You are an expert research assistant dedicated to helping users understand academic papers. You will be provided with the paper's content, annotations, relevant referenced papers, and pertinent web search results. Your task is to answer the user's questions, comments, or claims based solely on this information and your general knowledge.

    Instructions:
    - **Use only the provided sources:** If you include any information from the paper, referenced papers, or web search results, cite the source clearly.
    - **Consider existing annotations:** Reference and build upon previous annotations and their AI responses when relevant.
    - **Be precise and detailed:** Ensure your responses are comprehensive and directly address the user's input.
    - **Stay on topic:** Your responses should relate only to the content of the paper and the user's question.
    - **Avoid hallucinations:** If the information is not available in the provided content, explicitly state that you cannot answer based on the available information.
    - **Format using Markdown:** Structure your answer with Markdown to improve readability.

    Paper Excerpts (most relevant to the user question):
    {context_text}

    Existing Annotations and Their Responses:
    {annotations_context}

    Referenced Papers:
    {referenced_papers_context}

    Web Search Results:
    {web_context}

    User Question:
    {message}
    """
    
    try:
        api_key = _effective_api_key(provider, data)
        response = run_async(get_ai_response(provider, prompt, api_key=api_key))
        return jsonify({
            "response": response
        })
    except Exception as e:
        print("\n=== ERROR OCCURRED ===")
        print("Error type:", type(e).__name__)
        print("Error message:", str(e))
        return jsonify({
            "error": str(e)
        }), 500

@app.route("/ai-providers", methods=["GET"])
def get_ai_providers():
    # Return all supported providers; indicate whether server is configured with a key.
    return jsonify({
        "providers": [
            {
                "id": provider_id,
                "name": provider_info["name"],
                "enabled": bool(provider_info.get("enabled", True)),
                "serverConfigured": bool(provider_info.get("api_key"))
            }
            for provider_id, provider_info in AI_PROVIDERS.items()
        ]
    })

def create_upload_structure(upload_id):
    """Create directory structure for a new upload"""
    base_path = Path(app.config['STORAGE_BASE']) / upload_id
    pdf_path = base_path / 'pdf'
    figures_path = base_path / 'figures'
    chats_path = base_path / 'chats'
    citations_path = base_path / 'citations'   
    # Create directories
    pdf_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)
    chats_path.mkdir(parents=True, exist_ok=True)
    citations_path.mkdir(parents=True, exist_ok=True)
    
    return {
        'base_path': base_path,
        'pdf_path': pdf_path,
        'figures_path': figures_path,
        'chats_path': chats_path,
        'citations_path': citations_path  
    }

def save_metadata(upload_id, metadata):
    """Save metadata for an upload"""
    base_path = Path(app.config['STORAGE_BASE']) / upload_id
    metadata_file = base_path / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

def get_metadata(upload_id):
    """Get metadata for an upload"""
    base_path = Path(app.config['STORAGE_BASE']) / upload_id
    metadata_file = base_path / 'metadata.json'
    if metadata_file.exists():
        with open(metadata_file) as f:
            return json.load(f)
    return None

def get_chat_history(upload_id):
    """Get chat history for an upload"""
    base_path = Path(app.config['STORAGE_BASE']) / upload_id / 'chats'
    chat_file = base_path / 'history.json'
    
    # Create directory if it doesn't exist
    base_path.mkdir(parents=True, exist_ok=True)
    
    if chat_file.exists():
        with open(chat_file) as f:
            return json.load(f)
    # Initialize empty chat history file if it doesn't exist
    save_chat_history(upload_id, [])
    return []

def save_chat_history(upload_id, chat_history):
    """Save chat history for an upload"""
    base_path = Path(app.config['STORAGE_BASE']) / upload_id / 'chats'
    chat_file = base_path / 'history.json'
    
    # Ensure the chats directory exists
    base_path.mkdir(parents=True, exist_ok=True)
    
    with open(chat_file, 'w') as f:
        json.dump(chat_history, f, indent=2)

@app.route("/chat-history/<upload_id>", methods=["GET"])
def get_chat_history_endpoint(upload_id):
    history = get_chat_history(upload_id)
    return jsonify(history)

@app.route("/chat-history/<upload_id>", methods=["POST"])
def save_chat_history_endpoint(upload_id):
    history = request.get_json()
    save_chat_history(upload_id, history)
    return jsonify({"status": "ok"})

async def perform_web_search(query, num_results=3):
    try:
        search_results = []
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=num_results)
            for r in results:
                search_results.append(f"Source: {r['title']} ({r['link']})\n{r['body']}\n")
        
        return "\n".join(search_results)
    except Exception as e:
        print(f"Web search error: {str(e)}")
        return ""

async def get_ai_response(provider, prompt, temperature=0.7, api_key: str | None = None):
    logging.info("Getting AI response from provider: %s", provider)
    logging.debug("Provider supported: %s", AI_PROVIDERS.get(provider, {}).get('enabled', False))
    
    try:
        # Extract the actual content to search for
        search_query = ""
        if "verify" in prompt.lower():
            # Extract both the comment and selected text from the prompt
            comment_match = re.search(r'User\'s Comment/Claim: "(.*?)"', prompt)
            text_match = re.search(r'Selected Text: "(.*?)"', prompt)
            search_query = f"{comment_match.group(1) if comment_match else ''} {text_match.group(1) if text_match else ''}"
        elif "answer" in prompt.lower():
            # Extract both the question and selected text from the prompt
            question_match = re.search(r'Question: "(.*?)"', prompt)
            text_match = re.search(r'Selected Text: "(.*?)"', prompt)
            search_query = f"{question_match.group(1) if question_match else ''} {text_match.group(1) if text_match else ''}"
        
        # Clean up search query
        search_query = search_query.strip()
        if search_query:
            web_results = await perform_web_search(search_query)
            if web_results:
                prompt += f"\n\nRelevant information from web search:\n{web_results}\n\nPlease incorporate relevant web information in your response, citing sources when appropriate."
        
        if provider == 'gemini' and AI_PROVIDERS.get('gemini', {}).get('enabled', False):
            try:
                logging.info("Generating Gemini response...")
                key = api_key or AI_PROVIDERS.get("gemini", {}).get("api_key")
                if not key:
                    raise ValueError("No Gemini API key provided.")
                client = genai.Client(api_key=key)
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                    config={'temperature': temperature}
                )
                if not response.text:
                    return "Response blocked due to content safety filters."
                return response.text
            except Exception as e:
                logging.error("Gemini error details: %s", str(e))
                raise Exception(f"Gemini error: {str(e)}")
            
        elif provider == 'openai' and AI_PROVIDERS.get('openai', {}).get('enabled', False):
            try:
                logging.info("Generating OpenAI response...")
                key = api_key or AI_PROVIDERS.get("openai", {}).get("api_key")
                if not key:
                    raise ValueError("No OpenAI API key provided.")
                client = OpenAI(api_key=key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful research assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature
                )
                return response.choices[0].message.content
            except Exception as e:
                logging.error("OpenAI error details: %s", str(e))
                raise Exception(f"OpenAI error: {str(e)}")
            
        else:
            available_providers = [p for p in AI_PROVIDERS if AI_PROVIDERS[p]['enabled']]
            raise ValueError(f"AI provider '{provider}' not available. Available providers: {available_providers}")
            
    except Exception as e:
        logging.error("AI response error: %s", str(e))
        raise Exception(str(e))

def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

def _iter_gemini_stream(prompt: str, temperature: float = 0.7, api_key: str | None = None):
    if not AI_PROVIDERS.get('gemini', {}).get('enabled'):
        raise Exception("Gemini is not supported")
    key = api_key or AI_PROVIDERS.get("gemini", {}).get("api_key")
    if not key:
        raise Exception("No Gemini API key provided.")
    client = genai.Client(api_key=key)
    # Prefer streaming API when available
    stream_fn = getattr(client.models, "generate_content_stream", None)
    if callable(stream_fn):
        for chunk in stream_fn(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"temperature": temperature},
        ):
            text = getattr(chunk, "text", None)
            if text:
                yield text
        return
    # Fallback: single-shot response
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"temperature": temperature},
    )
    if resp and getattr(resp, "text", None):
        yield resp.text

def _iter_ai_stream(provider: str, prompt: str, temperature: float = 0.7, api_key: str | None = None):
    # For now, stream Gemini; OpenAI streaming can be added once enabled.
    if provider == "gemini":
        yield from _iter_gemini_stream(prompt, temperature=temperature, api_key=api_key)
    elif provider == "openai" and AI_PROVIDERS.get("openai", {}).get("enabled"):
        # Fallback to non-stream until OpenAI is enabled and verified.
        text = run_async(get_ai_response(provider, prompt, temperature=temperature, api_key=api_key))
        yield text
    else:
        text = run_async(get_ai_response(provider, prompt, temperature=temperature, api_key=api_key))
        yield text

@app.route("/chat-with-paper/stream", methods=["POST"])
def chat_with_paper_stream():
    data = request.get_json() or {}
    upload_id = data.get("uploadId")
    if not upload_id:
        return jsonify({"error": "Upload ID is required"}), 400

    provider = data.get("provider")
    if not provider or provider not in AI_PROVIDERS:
        return jsonify({"error": "Invalid AI provider"}), 400
    try:
        _ensure_provider_has_key(provider, data)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    api_key = _effective_api_key(provider, data)

    message = data.get("message", "")
    paper_context = data.get("paper_context", {})
    annotations_context = data.get("annotations_context", "")

    # Same retrieval behavior as non-streaming endpoint
    full_text_plain = _paper_context_to_plaintext(paper_context)
    digest = _text_digest(full_text_plain)
    with paper_document_stores_lock:
        cached = paper_document_stores.get(upload_id)
        if not cached or cached.get("digest") != digest:
            ds = DocumentStore()
            ds.add_document(full_text_plain, chunk_size=1200, overlap=4)
            paper_document_stores[upload_id] = {"digest": digest, "store": ds}
        doc_store = paper_document_stores[upload_id]["store"]

    try:
        relevant_chunks = doc_store.search(message, k=5) if full_text_plain else []
    except Exception:
        relevant_chunks = []
    context_text = "\n\n---\n\n".join(relevant_chunks) if relevant_chunks else (full_text_plain[:6000] if full_text_plain else "")

    referenced_papers_context = get_referenced_papers_context(upload_id)

    # Perform web search (single pass)
    try:
        web_results = run_async(perform_web_search(message))
        web_context = "\n\nRelevant information from web search:\n" + web_results if web_results else ""
    except Exception:
        web_context = ""

    prompt = f"""
    You are an expert research assistant dedicated to helping users understand academic papers. You will be provided with paper excerpts, annotations, relevant referenced papers, and pertinent web search results. Your task is to answer the user's questions based on this information and your general knowledge.

    Instructions:
    - **Use only the provided sources:** If you include any information from the paper, referenced papers, or web search results, cite the source clearly.
    - **Consider existing annotations:** Reference and build upon previous annotations and their AI responses when relevant.
    - **Be precise and detailed:** Ensure your responses are comprehensive and directly address the user's input.
    - **Avoid hallucinations:** If the information is not available in the provided content, explicitly state that.
    - **Format using Markdown:** Structure your answer with Markdown to improve readability.

    Paper Excerpts (most relevant to the user question):
    {context_text}

    Existing Annotations and Their Responses:
    {annotations_context}

    Referenced Papers:
    {referenced_papers_context}

    Web Search Results:
    {web_context}

    User Question:
    {message}
    """

    def generate():
        yield _sse_event("start", {"ok": True})
        full = ""
        try:
            for token in _iter_ai_stream(provider, prompt, temperature=0.7, api_key=api_key):
                full += token
                yield _sse_event("token", {"token": token})
            yield _sse_event("done", {"ok": True, "text": full})
        except Exception as e:
            yield _sse_event("error", {"error": str(e)})

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return Response(stream_with_context(generate()), mimetype="text/event-stream", headers=headers)

@app.route("/chat-multi/stream", methods=["POST"])
def chat_multi_stream():
    data = request.get_json() or {}
    provider = data.get("provider")
    if not provider or provider not in AI_PROVIDERS:
        return jsonify({"error": "Invalid AI provider"}), 400
    try:
        _ensure_provider_has_key(provider, data)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    api_key = _effective_api_key(provider, data)

    upload_ids = data.get("uploadIds") or []
    if not isinstance(upload_ids, list) or len(upload_ids) < 2:
        return jsonify({"error": "uploadIds must be a list with 2+ items"}), 400

    message = data.get("message", "")
    annotations_context = data.get("annotations_context", "")

    # Load stored conversations to get extraction payloads/titles
    convs = load_conversations()
    conv_map = {c.get("id"): c for c in convs if isinstance(c, dict) and c.get("id")}

    paper_blocks = []
    for uid in upload_ids:
        conv = conv_map.get(uid)
        if not conv:
            continue
        title = conv.get("title") or uid
        extraction = conv.get("extraction") or {}
        full_text_plain = _paper_context_to_plaintext(extraction)
        if not full_text_plain:
            continue

        digest = _text_digest(full_text_plain)
        with paper_document_stores_lock:
            cached = paper_document_stores.get(uid)
            if not cached or cached.get("digest") != digest:
                ds = DocumentStore()
                ds.add_document(full_text_plain, chunk_size=1200, overlap=4)
                paper_document_stores[uid] = {"digest": digest, "store": ds}
            doc_store = paper_document_stores[uid]["store"]

        try:
            chunks = doc_store.search(message, k=3)
        except Exception:
            chunks = []

        if chunks:
            paper_blocks.append(
                f"=== Paper: {title} ({uid}) ===\n" + "\n\n---\n\n".join(chunks)
            )

    if not paper_blocks:
        return jsonify({"error": "No papers found/loaded for the provided uploadIds"}), 400

    # Web search (single pass)
    try:
        web_results = run_async(perform_web_search(message))
        web_context = "\n\nRelevant information from web search:\n" + web_results if web_results else ""
    except Exception:
        web_context = ""

    prompt = f"""
    You are an expert research assistant. Answer the user's question by synthesizing evidence across multiple papers.

    Instructions:
    - Cite which paper a statement comes from using the paper title if possible.
    - If papers disagree, call it out explicitly.
    - Avoid hallucinations; if not supported by provided excerpts, say so.
    - Format using Markdown.

    Paper Excerpts (retrieved by relevance):
    {chr(10).join(paper_blocks)}

    Existing Annotations and Their Responses (current workspace):
    {annotations_context}

    Web Search Results:
    {web_context}

    User Question:
    {message}
    """

    def generate():
        yield _sse_event("start", {"ok": True})
        full = ""
        try:
            for token in _iter_ai_stream(provider, prompt, temperature=0.7, api_key=api_key):
                full += token
                yield _sse_event("token", {"token": token})
            yield _sse_event("done", {"ok": True, "text": full})
        except Exception as e:
            yield _sse_event("error", {"error": str(e)})

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return Response(stream_with_context(generate()), mimetype="text/event-stream", headers=headers)

@app.route("/verify-annotation/stream", methods=["POST"])
def verify_annotation_stream():
    data = request.get_json() or {}
    provider = data.get("provider")
    upload_id = data.get("uploadId")

    if not upload_id:
        return jsonify({"error": "Upload ID is required"}), 400
    if not provider or provider not in AI_PROVIDERS:
        return jsonify({"error": "Invalid AI provider"}), 400
    try:
        _ensure_provider_has_key(provider, data)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    api_key = _effective_api_key(provider, data)

    annotation_text = data.get("selectedText", "")
    comment = data.get("comment", "")
    context = data.get("context", "")
    figures = data.get("figures", [])

    # Web search for relevant information
    try:
        web_results = run_async(perform_web_search(f"{comment} {annotation_text}"))
        web_context = "\n\nRelevant information from web search:\n" + web_results if web_results else ""
    except Exception:
        web_context = ""

    figures_context = "\n".join(
        [
            f"Figure {fig['ref']} (page {fig.get('page_number')}): {fig.get('caption') or 'No caption found.'}"
            for fig in figures if fig.get("ref")
        ]
    )
    referenced_papers_context = get_referenced_papers_context(upload_id)

    prompt = f"""
    You are an expert AI research assistant. Your task is to verify the accuracy of a user's comment or claim regarding a specific annotation in a paper. Use only the provided sources—selected text, context from the paper, referenced papers, available figures, and web search results—along with any relevant general knowledge.

    **Inputs Provided:**
    - **Selected Text:** "{annotation_text}"
    - **User's Comment/Claim:** "{comment}"
    - **Context from the Paper:**  
    {context}
    - **Available Figures in Context:**  
    {figures_context}
    - **Referenced Papers Context:**  
    {referenced_papers_context}
    - **Web Search Results:**  
    {web_context}

    **Instructions:**
    1. **Verification:** Verify whether the user's comment/claim about the selected text is accurate.
    2. **Evidence from the Paper:** Quote and cite specific parts of the paper.
    3. **Evidence from Referenced Papers:** Quote and cite specific parts from referenced papers.
    4. **Evidence from Web Sources:** Cite web sources as [Title](URL) when available.
    5. **General Knowledge:** Indicate relevant general knowledge.
    6. **Suggested Corrections:** Provide corrections if applicable.

    Structure your response using Markdown for readability. Avoid hallucinations; if insufficient info, say so.
    """

    def generate():
        yield _sse_event("start", {"ok": True})
        full = ""
        try:
            for token in _iter_ai_stream(provider, prompt, temperature=0.7, api_key=api_key):
                full += token
                yield _sse_event("token", {"token": token})
            yield _sse_event("done", {"ok": True, "text": full})
        except Exception as e:
            yield _sse_event("error", {"error": str(e)})

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return Response(stream_with_context(generate()), mimetype="text/event-stream", headers=headers)

@app.route("/answer-question/stream", methods=["POST"])
def answer_question_stream():
    data = request.get_json() or {}
    upload_id = data.get("uploadId")
    if not upload_id:
        return jsonify({"error": "Upload ID is required"}), 400

    provider = data.get("provider")
    if not provider or provider not in AI_PROVIDERS:
        return jsonify({"error": "Invalid AI provider"}), 400
    try:
        _ensure_provider_has_key(provider, data)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    api_key = _effective_api_key(provider, data)

    question = data.get("question", "")
    context = data.get("context", "")
    selected_text = data.get("selectedText", "")
    figures = data.get("figures", [])

    # Retrieval (same approach as /answer-question)
    with document_stores_lock:
        if upload_id not in document_stores:
            doc_store = DocumentStore()
            doc_store.add_document(context)
            document_stores[upload_id] = doc_store
        relevant_chunks = document_stores[upload_id].search(question, k=3)
    context_text = "\n\n".join(relevant_chunks)

    figures_context = "\n".join(
        [
            f"Figure {fig['ref']} (page {fig.get('page_number')}): {fig.get('caption') or 'No caption found.'}"
            for fig in figures if fig.get("ref")
        ]
    )
    referenced_papers_context = get_referenced_papers_context(upload_id)

    prompt = f"""
    You are an expert AI research assistant tasked with answering questions about a specific annotated part of a paper.

    **Inputs Provided:**
    - **Question:** "{question}"
    - **Selected Text:** "{selected_text}"
    - **Context from the Paper:**  
    {context_text}
    - **Available Figures in Context:**  
    {figures_context}
    - **Referenced Papers Context:**  
    {referenced_papers_context}

    Include sections: Answer, Evidence from the Paper, Evidence from Referenced Papers, General Knowledge, Additional Context.
    Format using Markdown. If insufficient info, say so.
    """

    def generate():
        yield _sse_event("start", {"ok": True})
        full = ""
        try:
            for token in _iter_ai_stream(provider, prompt, temperature=0.7, api_key=api_key):
                full += token
                yield _sse_event("token", {"token": token})
            yield _sse_event("done", {"ok": True, "text": full})
        except Exception as e:
            yield _sse_event("error", {"error": str(e)})

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return Response(stream_with_context(generate()), mimetype="text/event-stream", headers=headers)

def store_cited_paper(metadata, full_text, upload_id, db_path="cited_papers.db"):
    """Store cited paper metadata and full text into the local SQLite database with its associated upload_id."""
    doi = metadata.get("DOI", "") or ""
    arxiv_id = metadata.get("arxiv_id", "") or ""
    title_list = metadata.get("title", [])
    title = title_list[0] if title_list else ""
    publisher = metadata.get("publisher", "") or ""
    year = metadata.get("year")
    raw_text = metadata.get("raw_text", "") or ""
    authors_data = metadata.get("author", [])
    # Handle authors as either a string or list of dicts
    if isinstance(authors_data, str):
        authors = authors_data
    elif isinstance(authors_data, list) and authors_data:
        authors = ", ".join([
            f"{a.get('given', '').strip()} {a.get('family', '').strip()}"
            for a in authors_data if isinstance(a, dict)
        ])
    else:
        authors = ""
    abstract = metadata.get("abstract", "") or ""
    # Determine the URL from "link" if available or fallback to "URL".
    url = ""
    links = metadata.get("link", [])
    if isinstance(links, list) and len(links) > 0:
        first_link = links[0]
        if isinstance(first_link, dict):
            url = first_link.get("URL", "")
    if not url:
        url = metadata.get("URL", "") or ""
    
    # Need at least a title or some identifier to store
    if not doi and not arxiv_id and not title and not raw_text:
        logging.warning("Skipping paper without any identifier, title, or raw text")
        return
    
    # Generate a unique key for deduplication
    # Use DOI > arXiv ID > title hash > raw_text hash as the unique identifier
    if doi:
        unique_key = f"doi:{doi}"
    elif arxiv_id:
        unique_key = f"arxiv:{arxiv_id}"
    elif title:
        unique_key = f"title:{hash(title)}"
    else:
        unique_key = f"raw:{hash(raw_text[:200])}"
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    try:
        c.execute('''
            INSERT OR REPLACE INTO cited_papers (unique_key, doi, arxiv_id, upload_id, title, publisher, authors, abstract, year, full_text, url, raw_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (unique_key, doi, arxiv_id, upload_id, title, publisher, authors, abstract, year, full_text, url, raw_text))
        conn.commit()
        identifier = doi or arxiv_id or (title[:40] if title else raw_text[:40])
        logging.info("Stored cited paper: %s for upload_id: %s", identifier, upload_id)
    except Exception as e:
        logging.error("Error inserting into DB: %s", str(e))
    finally:
        conn.close()

def fetch_full_text(url):
    """Fetch and extract text content from the given URL using BeautifulSoup."""
    try:
        response = httpx.get(url, timeout=10.0)
        if response.status_code == 200:
            html = response.text
            soup = BeautifulSoup(html, "html.parser")
            # Remove unwanted tags
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator="\n")
            # Simplify whitespace
            text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
            logging.info("Fetched full text from %s", url)
            return text
        else:
            logging.warning("Failed to fetch full text from %s, status %s", url, response.status_code)
            return None
    except Exception as e:
        logging.error("Exception fetching full text from %s: %s", url, str(e))
        return None

# After configuring AI_PROVIDERS, add detailed debug logging:
print("\nInitialized AI Providers:")
for provider_id, info in AI_PROVIDERS.items():
    print(f"{provider_id}: enabled = {info['enabled']}, api_key = {'set' if info['api_key'] else 'not set'}")

def init_cited_papers_db(db_path="cited_papers.db"):
    """Initialize the SQLite database to store cited paper content."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Create table with a unique_key column for proper deduplication
    c.execute('''
        CREATE TABLE IF NOT EXISTS cited_papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            unique_key TEXT,
            doi TEXT,
            arxiv_id TEXT,
            upload_id TEXT,
            title TEXT,
            publisher TEXT,
            authors TEXT,
            abstract TEXT,
            year INTEGER,
            full_text TEXT,
            url TEXT,
            raw_text TEXT,
            UNIQUE(unique_key, upload_id)
        )
    ''')
    
    # Add columns if they don't exist (for existing databases)
    for column, coltype in [('arxiv_id', 'TEXT'), ('year', 'INTEGER'), ('raw_text', 'TEXT'), ('unique_key', 'TEXT')]:
        try:
            c.execute(f'ALTER TABLE cited_papers ADD COLUMN {column} {coltype}')
        except sqlite3.OperationalError:
            pass  # Column already exists
    
    # Create index for faster lookups
    try:
        c.execute('CREATE INDEX IF NOT EXISTS idx_upload_id ON cited_papers(upload_id)')
    except sqlite3.OperationalError:
        pass
    
    conn.commit()
    conn.close()
    logging.info("Initialized cited papers database at %s", db_path)

def delete_cited_papers(upload_id, db_path="cited_papers.db"):
    """Delete all cited papers in the database that are linked with the given upload_id."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    try:
        c.execute("DELETE FROM cited_papers WHERE upload_id = ?", (upload_id,))
        conn.commit()
        logging.info("Deleted cited papers for upload_id: %s", upload_id)
    except Exception as e:
        logging.error("Error deleting cited papers for upload_id %s: %s", upload_id, str(e))
    finally:
        conn.close()

def process_references_background(pdf_path, upload_id, citations_path, task):
    try:
        # Update initial progress
        task.progress = 10
        task.status = "Extracting references from PDF..."
        
        # Extract references (returns list of dicts with type, id, query)
        references = extract_references_from_pdf(pdf_path)
        task.progress = 30
        
        total_refs = len(references) if references else 0
        processed_refs = 0
        citations = []
        
        logging.info("Found %d references to process", total_refs)
        
        for ref_info in references:
            # Update progress based on processed references
            processed_refs += 1
            if total_refs > 0:
                task.progress = 30 + int((processed_refs / total_refs) * 60)
            else:
                task.progress = 90
            
            ref_type = ref_info.get('type', 'unknown')
            ref_id = ref_info.get('id', '')
            raw_text = ref_info.get('raw_text', '')
            task.status = f"Processing reference {processed_refs}/{total_refs}: {ref_type}"
            
            try:
                # Fetch paper metadata using Semantic Scholar, OpenAlex, or Crossref
                paper_metadata = fetch_paper_metadata(ref_info, UNPAYWALL_EMAIL)
            except Exception as e:
                logging.warning("API error for reference %s: %s", ref_id[:30], str(e))
                paper_metadata = None
            
            # Even if lookup fails, store what we have from extraction
            if not paper_metadata:
                logging.info("Using extracted data for reference: %s", ref_id[:50])
                paper_metadata = {
                    'title': ref_info.get('title') or ref_info.get('query', ref_id)[:200],
                    'authors': ref_info.get('authors', ''),
                    'abstract': '',
                    'year': ref_info.get('year'),
                    'doi': ref_id if ref_type == 'doi' else None,
                    'arxiv_id': ref_id if ref_type == 'arxiv' else None,
                    'url': '',
                    'source': 'extracted'
                }
            
            # Get DOI for Unpaywall lookup
            doi = paper_metadata.get('doi') or (ref_id if ref_type == 'doi' else None)
            
            # Try to get full text URL
            oa_pdf_url = None
            full_text = None
            
            # Check for arXiv URL first (most reliable for CS papers)
            arxiv_id = paper_metadata.get('arxiv_id') or (ref_id if ref_type == 'arxiv' else None)
            if arxiv_id:
                oa_pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
            # Try Unpaywall for DOI-based papers
            if not oa_pdf_url and doi:
                try:
                    oa_pdf_url = get_open_access_pdf_url(doi, UNPAYWALL_EMAIL)
                except Exception as e:
                    logging.debug("Unpaywall lookup failed for %s: %s", doi, str(e))
            
            # Build metadata record - ALWAYS store the reference
            metadata = {
                "DOI": doi or "",
                "arxiv_id": arxiv_id or "",
                "title": [paper_metadata.get("title", ref_id)],
                "publisher": paper_metadata.get("publisher", ""),
                "author": paper_metadata.get("authors", ""),
                "abstract": paper_metadata.get("abstract", ""),
                "year": paper_metadata.get("year"),
                "link": [{"URL": oa_pdf_url}] if oa_pdf_url else [],
                "URL": oa_pdf_url or paper_metadata.get("url", ""),
                "raw_text": raw_text[:500] if raw_text else ""  # Store raw text as fallback
            }
            store_cited_paper(metadata, full_text, upload_id)
            citations.append(metadata)
        
        # Save citations
        citations_file = citations_path / "citations.json"
        with open(citations_file, "w") as f:
            json.dump(citations, f, indent=2)
        
        # Update task completion
        task.progress = 100
        task.status = f"Complete - {len(citations)} references processed"
        task.result = citations
        logging.info("Reference processing complete: %d citations stored", len(citations))
        
    except Exception as e:
        logging.error("Error in background processing: %s", str(e))
        task.status = "Error"
        task.error = str(e)
        task.progress = 100

@app.route('/task-progress/<task_id>')
def task_progress(task_id):
    with background_tasks_lock:
        task = background_tasks.get(task_id)
    
    if not task:
        return jsonify({"error": "Task not found"}), 404
    
    response = {
        "progress": task.progress,
        "status": task.status,
        "complete": task.progress == 100,
        "error": task.error,
        "result": task.result
    }
    
    # Clean up completed tasks after client has retrieved the result
    if task.progress == 100:
        with background_tasks_lock:
            if task_id in background_tasks:
                del background_tasks[task_id]
                logging.info("Cleaned up completed background task: %s", task_id)
    
    return jsonify(response)

@app.route('/citations/<upload_id>', methods=['GET'])
def get_citations(upload_id):
    """Get all citations/references for a given upload."""
    try:
        with sqlite3.connect("cited_papers.db") as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT doi, arxiv_id, title, authors, abstract, year, url, raw_text
                FROM cited_papers
                WHERE upload_id = ?
                ORDER BY year DESC, title
            """, (upload_id,))
            
            papers = cursor.fetchall()
            
            citations = []
            for doi, arxiv_id, title, authors, abstract, year, url, raw_text in papers:
                # Create identifier for display
                identifier = doi or (f"arXiv:{arxiv_id}" if arxiv_id else None)
                
                # Use raw_text as fallback for title if no proper title
                display_title = title
                if not title or title == doi or title == arxiv_id:
                    display_title = raw_text[:200] if raw_text else None
                
                citations.append({
                    "doi": doi,
                    "arxiv_id": arxiv_id,
                    "identifier": identifier,
                    "title": display_title,
                    "authors": authors,
                    "abstract": abstract,
                    "year": year,
                    "url": url,
                    "raw_text": raw_text,
                    "hasFullText": bool(url)
                })
            
            return jsonify({"citations": citations, "count": len(citations)})
    except Exception as e:
        logging.error("Error fetching citations for upload_id %s: %s", upload_id, str(e))
        return jsonify({"error": str(e)}), 500

def _get_citation_for_summary(upload_id: str, doi: str = "", arxiv_id: str = "", title: str = "") -> dict:
    """Fetch a single citation record for summarization."""
    doi = (doi or "").strip()
    arxiv_id = (arxiv_id or "").strip()
    title = (title or "").strip()

    with sqlite3.connect("cited_papers.db") as conn:
        cursor = conn.cursor()

        if doi:
            cursor.execute(
                """
                SELECT doi, arxiv_id, title, authors, abstract, year, full_text, url, raw_text
                FROM cited_papers
                WHERE upload_id = ? AND doi = ?
                LIMIT 1
                """,
                (upload_id, doi),
            )
        elif arxiv_id:
            cursor.execute(
                """
                SELECT doi, arxiv_id, title, authors, abstract, year, full_text, url, raw_text
                FROM cited_papers
                WHERE upload_id = ? AND arxiv_id = ?
                LIMIT 1
                """,
                (upload_id, arxiv_id),
            )
        elif title:
            cursor.execute(
                """
                SELECT doi, arxiv_id, title, authors, abstract, year, full_text, url, raw_text
                FROM cited_papers
                WHERE upload_id = ? AND title = ?
                LIMIT 1
                """,
                (upload_id, title),
            )
        else:
            return {}

        row = cursor.fetchone()
        if not row:
            return {}
        doi, arxiv_id, title, authors, abstract, year, full_text, url, raw_text = row
        return {
            "doi": doi or "",
            "arxiv_id": arxiv_id or "",
            "title": title or "",
            "authors": authors or "",
            "abstract": abstract or "",
            "year": year,
            "full_text": full_text or "",
            "url": url or "",
            "raw_text": raw_text or "",
        }

@app.route("/summarize-citation", methods=["POST"])
def summarize_citation():
    data = request.get_json() or {}
    upload_id = data.get("uploadId")
    if not upload_id:
        return jsonify({"error": "Upload ID is required"}), 400

    provider = data.get("provider")
    if not provider or provider not in AI_PROVIDERS:
        return jsonify({"error": "Invalid AI provider"}), 400
    try:
        _ensure_provider_has_key(provider, data)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    citation = _get_citation_for_summary(
        upload_id,
        doi=data.get("doi", ""),
        arxiv_id=data.get("arxiv_id", ""),
        title=data.get("title", ""),
    )
    if not citation:
        return jsonify({"error": "Citation not found"}), 404

    prompt = f"""
    You are an expert research assistant. Summarize the cited paper below using only the provided metadata/text.

    Title: {citation.get('title','')}
    Authors: {citation.get('authors','')}
    Year: {citation.get('year','')}
    DOI: {citation.get('doi','')}
    arXiv: {citation.get('arxiv_id','')}
    URL: {citation.get('url','')}

    Abstract:
    {citation.get('abstract','')}

    Extracted Reference Text:
    {citation.get('raw_text','')}

    If full text excerpt is present, use it; otherwise say it's not available.
    Full Text (may be empty):
    {citation.get('full_text','')[:2000]}

    Output in Markdown with sections:
    - Summary (3-6 sentences)
    - Key contributions (bullets)
    - Methods / approach (bullets)
    - When to cite this (bullets)
    - Limitations / caveats (bullets)
    """

    try:
        api_key = _effective_api_key(provider, data)
        summary = run_async(get_ai_response(provider, prompt, api_key=api_key))
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/summarize-citation/stream", methods=["POST"])
def summarize_citation_stream():
    data = request.get_json() or {}
    upload_id = data.get("uploadId")
    if not upload_id:
        return jsonify({"error": "Upload ID is required"}), 400

    provider = data.get("provider")
    if not provider or provider not in AI_PROVIDERS:
        return jsonify({"error": "Invalid AI provider"}), 400
    try:
        _ensure_provider_has_key(provider, data)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    api_key = _effective_api_key(provider, data)

    citation = _get_citation_for_summary(
        upload_id,
        doi=data.get("doi", ""),
        arxiv_id=data.get("arxiv_id", ""),
        title=data.get("title", ""),
    )
    if not citation:
        return jsonify({"error": "Citation not found"}), 404

    prompt = f"""
    You are an expert research assistant. Summarize the cited paper below using only the provided metadata/text.

    Title: {citation.get('title','')}
    Authors: {citation.get('authors','')}
    Year: {citation.get('year','')}
    DOI: {citation.get('doi','')}
    arXiv: {citation.get('arxiv_id','')}
    URL: {citation.get('url','')}

    Abstract:
    {citation.get('abstract','')}

    Extracted Reference Text:
    {citation.get('raw_text','')}

    Full Text (may be empty):
    {citation.get('full_text','')[:2000]}

    Output in Markdown with sections:
    - Summary (3-6 sentences)
    - Key contributions (bullets)
    - Methods / approach (bullets)
    - When to cite this (bullets)
    - Limitations / caveats (bullets)
    """

    def generate():
        yield _sse_event("start", {"ok": True})
        full = ""
        try:
            for token in _iter_ai_stream(provider, prompt, temperature=0.7, api_key=api_key):
                full += token
                yield _sse_event("token", {"token": token})
            yield _sse_event("done", {"ok": True, "text": full})
        except Exception as e:
            yield _sse_event("error", {"error": str(e)})

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return Response(stream_with_context(generate()), mimetype="text/event-stream", headers=headers)

def get_referenced_papers_context(conversation_id):
    """Fetch context from referenced papers for a given conversation."""
    try:
        # Connect to your SQLite database
        with sqlite3.connect("cited_papers.db") as conn:
            cursor = conn.cursor()
            
            # Query to get referenced papers for the conversation
            cursor.execute("""
                SELECT doi, arxiv_id, title, authors, abstract, year, full_text, url
                FROM cited_papers
                WHERE upload_id = ?
            """, (conversation_id,))
            
            papers = cursor.fetchall()
            
            if not papers:
                return ""
            
            # Format the context
            context = f"\n=== {len(papers)} Referenced Papers ===\n"
            for doi, arxiv_id, title, authors, abstract, year, full_text, url in papers:
                context += f"\n--- Referenced Paper ---"
                if doi:
                    context += f"\nDOI: {doi}"
                if arxiv_id:
                    context += f"\narXiv: {arxiv_id}"
                if title and title != doi and title != arxiv_id:
                    context += f"\nTitle: {title}"
                if authors:
                    context += f"\nAuthors: {authors}"
                if year:
                    context += f"\nYear: {year}"
                if abstract:
                    context += f"\nAbstract: {abstract}"
                if full_text:
                    excerpt = full_text[:500] + "..." if len(full_text) > 500 else full_text
                    context += f"\nContent Excerpt: {excerpt}"
                if url:
                    context += f"\nURL: {url}"
                context += "\n"
            
            return context
    except Exception as e:
        logging.error(f"Error fetching referenced papers context: {str(e)}")
        return ""

@app.route("/explain-math", methods=["POST"])
def explain_math():
    data = request.get_json() or {}
    provider = data.get("provider")
    upload_id = data.get("uploadId")
    
    if not upload_id:
        return jsonify({"error": "Upload ID is required"}), 400
        
    if not provider or provider not in AI_PROVIDERS:
        return jsonify({"error": "Invalid AI provider"}), 400
    try:
        _ensure_provider_has_key(provider, data)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
    math_expression = data.get("expression", "")
    context = data.get("context", "")
    paper_context = data.get("paper_context", {})
    
    # Get referenced papers context
    referenced_papers_context = get_referenced_papers_context(upload_id)
    
    prompt = f"""
    As an AI research assistant, please explain the following mathematical expression in detail:
    
    Mathematical Expression: {math_expression}
    
    Context from the paper:
    {context}
    
    Referenced Papers Context:
    {referenced_papers_context}
    
    Please structure your response in the following sections:
    1. Basic Explanation: Explain what this mathematical expression represents in simple terms
    2. Variable Definitions: Define each variable and symbol used in the expression
    3. Mathematical Concepts: Explain the key mathematical concepts involved
    4. Relationship to Paper: Explain how this expression relates to the paper's content and findings
    5. Practical Applications: Describe any practical applications or implications
    6. Related Equations: Mention any related equations or mathematical concepts from the paper
    
    Format your response using markdown for better readability.
    Use LaTeX notation for mathematical expressions where appropriate, enclosed in $ symbols.
    """
    
    try:
        api_key = _effective_api_key(provider, data)
        response = run_async(get_ai_response(provider, prompt, api_key=api_key))
        return jsonify({"explanation": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/explain-math/stream", methods=["POST"])
def explain_math_stream():
    data = request.get_json() or {}
    provider = data.get("provider")
    upload_id = data.get("uploadId")

    if not upload_id:
        return jsonify({"error": "Upload ID is required"}), 400
    if not provider or provider not in AI_PROVIDERS:
        return jsonify({"error": "Invalid AI provider"}), 400
    try:
        _ensure_provider_has_key(provider, data)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    api_key = _effective_api_key(provider, data)

    math_expression = data.get("expression", "")
    context = data.get("context", "")

    referenced_papers_context = get_referenced_papers_context(upload_id)

    prompt = f"""
    As an AI research assistant, please explain the following mathematical expression in detail:
    
    Mathematical Expression: {math_expression}
    
    Context from the paper:
    {context}
    
    Referenced Papers Context:
    {referenced_papers_context}
    
    Please structure your response in the following sections:
    1. Basic Explanation
    2. Variable Definitions
    3. Mathematical Concepts
    4. Relationship to Paper
    5. Practical Applications
    6. Related Equations
    
    Format your response using markdown for better readability.
    Use LaTeX notation for mathematical expressions where appropriate, enclosed in $ symbols.
    """

    def generate():
        yield _sse_event("start", {"ok": True})
        full = ""
        try:
            for token in _iter_ai_stream(provider, prompt, temperature=0.7, api_key=api_key):
                full += token
                yield _sse_event("token", {"token": token})
            yield _sse_event("done", {"ok": True, "text": full})
        except Exception as e:
            yield _sse_event("error", {"error": str(e)})

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return Response(stream_with_context(generate()), mimetype="text/event-stream", headers=headers)

if __name__ == '__main__':
    init_cited_papers_db()
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, port=int(os.getenv("PORT", "5000")))