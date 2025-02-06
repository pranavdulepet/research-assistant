from flask import Flask, request, render_template, jsonify
import os
import fitz 
import re
import uuid
import math
import subprocess
import json
from dotenv import load_dotenv
import google.generativeai as genai
import shutil
from pathlib import Path
from datetime import datetime
import openai
import httpx
from openai import OpenAI
from duckduckgo_search import DDGS

app = Flask(__name__)

# Directories for uploads and extracted figures.
app.config['UPLOAD_FOLDER'] = 'static/figures'
app.config['UPLOAD_PDF'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['UPLOAD_PDF'], exist_ok=True)

# File where conversation data will be stored.
CONVERSATIONS_FILE = "conversations.json"

SPECIAL_THRESHOLD = 300  # threshold (in PDF points) for converting a figure to SVG

# Update the configuration to use base directories
app.config['STORAGE_BASE'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

load_dotenv()

# After load_dotenv(), add these debug prints:
print("\nEnvironment Variables Check:")
print(f"GEMINI_API_KEY present: {bool(os.getenv('GEMINI_API_KEY'))}")
print(f"OPENAI_API_KEY present: {bool(os.getenv('OPENAI_API_KEY'))}")

# Configure AI Providers
AI_PROVIDERS = {
    'gemini': {
        'id': 'gemini',
        'name': 'Google Gemini',
        'enabled': bool(os.getenv("GEMINI_API_KEY")),
        'api_key': os.getenv("GEMINI_API_KEY")
    },
    'openai': {
        'id': 'openai',
        'name': 'OpenAI GPT-4',
        'enabled': bool(os.getenv("OPENAI_API_KEY")),
        'api_key': os.getenv("OPENAI_API_KEY")
    },
    'deepseek': {
        'id': 'deepseek',
        'name': 'DeepSeek (Coming Soon)',
        'enabled': False,
        'api_key': None
    }
}

# Initialize AI clients
if AI_PROVIDERS['gemini']['enabled']:
    genai.configure(api_key=AI_PROVIDERS['gemini']['api_key'])
    gemini_model = genai.GenerativeModel('gemini-pro')
    print("Gemini model initialized successfully")

# if AI_PROVIDERS['openai']['enabled']:
#     openai_client = OpenAI(
#         api_key=AI_PROVIDERS['openai']['api_key']
#     )
#     print("OpenAI client initialized successfully, without proxy.")

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
    
    # Create unique ID and directory structure
    upload_id = str(uuid.uuid4())
    paths = create_upload_structure(upload_id)
    
    # Save the PDF
    pdf_path = paths['pdf_path'] / 'original.pdf'
    pdf_file.save(pdf_path)
    
    # Extract data
    extraction_data = extract_text_and_figures(str(pdf_path), upload_id, paths['figures_path'])
    
    # Save metadata
    metadata = {
        'upload_id': upload_id,
        'original_filename': pdf_file.filename,
        'upload_date': datetime.now().isoformat(),
        'extraction_data': extraction_data
    }
    save_metadata(upload_id, metadata)
    
    result = {
        "upload_id": upload_id,
        "pdf_url": f"/static/uploads/{upload_id}/pdf/original.pdf",
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

def extract_text_and_figures(pdf_path, upload_id, figures_path):
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
            width = bbox.width
            height = bbox.height
            bbox_center = ((bbox.x0 + bbox.x1) / 2.0, (bbox.y0 + bbox.y1) / 2.0)

            if width >= SPECIAL_THRESHOLD or height >= SPECIAL_THRESHOLD:
                filename = f"{uuid.uuid4()}.svg"
                output_svg_path = figures_path / filename
                convert_region_to_svg(pdf_path, page_index + 1, (bbox.x0, bbox.y0, bbox.x1, bbox.y1), str(output_svg_path))
                figure_url = f"/static/uploads/{upload_id}/figures/{filename}"
            else:
                base_image = doc.extract_image(xref)
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

@app.route("/delete-upload/<upload_id>", methods=["POST"])
def delete_upload(upload_id):
    try:
        # Get the base path for this upload
        base_path = Path(app.config['STORAGE_BASE']) / upload_id
        
        if base_path.exists():
            # Remove the entire directory structure for this upload
            shutil.rmtree(base_path)
            
            # Load and update conversations.json
            conversations = load_conversations()
            conversations = [c for c in conversations if c['id'] != upload_id]
            save_conversations(conversations)
            
            return jsonify({"status": "ok"})
        else:
            return jsonify({"error": "Upload not found"}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/verify-annotation", methods=["POST"])
def verify_annotation():
    data = request.get_json()
    provider = data.get("provider")  # Get provider from request
    
    if not provider or provider not in AI_PROVIDERS:
        return jsonify({"error": "Invalid AI provider"}), 400
    
    if not AI_PROVIDERS[provider]['enabled']:
        return jsonify({"error": f"{AI_PROVIDERS[provider]['name']} is not enabled"}), 400
    
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
    As an AI research assistant, please verify the following annotation based on the Context from the paper, the Available Figures in Context, and web search results:

    Selected Text: "{annotation_text}"
    User's Comment/Claim: "{comment}"

    Context from the paper:
    {context}

    Available Figures in Context:
    {figures_context}

    Please structure your response in the following sections:
    1. Verification: Verify if the comment/claim about the selected text is accurate
    2. Evidence from Paper: Quote and cite specific parts of the paper that support your verification
    3. Evidence from Web Sources: Clearly cite any web sources used, including URLs when available
    4. General Knowledge: Clearly indicate any additional information from your training
    5. Suggested Corrections: If needed, provide any corrections or clarifications

    Format your response using markdown for better readability.
    When citing web sources, please use the format: [Source Title](URL)
    """
    
    print("\n=== SENDING REQUEST TO GEMINI ===")
    try:
        print("Waiting for Gemini response...")
        # Use asyncio.run() to handle the async call in sync context
        import asyncio
        response = asyncio.run(get_ai_response(provider, prompt))
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

@app.route("/answer-question", methods=["POST"])
def answer_question():
    try:
        data = request.get_json()
        if not data:
            raise ValueError("No data received")
            
        provider = data.get("provider")
        if not provider or provider not in AI_PROVIDERS:
            return jsonify({"error": "Invalid AI provider"}), 400
        
        if not AI_PROVIDERS[provider]['enabled']:
            return jsonify({"error": f"{AI_PROVIDERS[provider]['name']} is not enabled"}), 400
        
        question = data.get("question", "")
        context = data.get("context", "")
        selected_text = data.get("selectedText", "")
        figures = data.get("figures", [])
        
        print(f"Processing answer for question: '{question}'")
        print(f"Selected provider: {provider}")
        print(f"Number of figures in context: {len(figures)}")
        
        # Build context with figures
        figures_context = "\n".join([
            f"Figure {fig['ref']}: Located at page {fig['page_number']}" 
            for fig in figures if fig.get('ref')
        ])
        
        prompt = f"""
        As an AI research assistant, please answer the following question using the Selected Text, Context from the paper, Available Figures in Context, and web search results:
        
        Question: "{question}"
        
        Selected Text: "{selected_text}"
        
        Context from the paper:
        {context}
        
        Available Figures in Context:
        {figures_context}
        
        Please structure your response in the following sections:
        1. Answer: Provide a clear and detailed answer
        2. Evidence from Paper: Quote and cite specific parts of the paper that support your answer
        3. Evidence from Web Sources: Clearly cite any web sources used, including URLs when available
        4. General Knowledge: Clearly indicate any additional information from your training
        5. Additional Context: Include any relevant figures or sections that might be helpful
        
        Format your response using markdown for better readability.
        When citing web sources, please use the format: [Source Title](URL)
        """
        
        try:
            # For synchronous operation, we'll use asyncio.run()
            import asyncio
            response = asyncio.run(get_ai_response(provider, prompt))
            return jsonify({"answer": response})
        except Exception as e:
            print(f"Error getting AI response: {str(e)}")
            return jsonify({"error": str(e)}), 500
            
    except Exception as e:
        print(f"Error in answer_question route: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@app.route("/chat-with-paper", methods=["POST"])
def chat_with_paper():
    data = request.get_json()
    provider = data.get("provider")
    
    if not provider or provider not in AI_PROVIDERS:
        return jsonify({"error": "Invalid AI provider"}), 400
    
    if not AI_PROVIDERS[provider]['enabled']:
        return jsonify({"error": f"{AI_PROVIDERS[provider]['name']} is not enabled"}), 400
    
    message = data.get("message", "")
    print("WEBSEARCH MESSAGE: ", message)
    paper_context = data.get("paper_context", {})
    
    # Combine all page text for context
    full_text = "\n".join([page.get("text", "") for page in paper_context.get("pages", [])])
    
    # Perform web search for relevant information
    try:
        # Use asyncio.run() to handle the async call in sync context
        import asyncio
        web_results = asyncio.run(perform_web_search(message))
        web_context = "\n\nRelevant information from web search:\n" + web_results if web_results else ""
    except Exception as e:
        print(f"Web search error: {str(e)}")
        web_context = ""
    
    prompt = f"""
    You are an AI research assistant helping to answer questions about a research paper which is given to you in the Paper content.
    Use the following paper content and web search results to answer the user's question.
    If you use information from web sources, clearly cite them in your response.
    If you cannot answer based on the available information, say so.
    
    Paper content:
    {full_text}
    {web_context}
    
    User question: {message}
    
    Please structure your response in the following paragraphs:
    Answer: Provide a clear and detailed answer
    Paper Evidence: Quote and cite specific parts of the paper that support your answer
    Web Sources: If web search results were used, cite the sources and how they contributed
    General Knowledge: Clearly indicate any additional information from your training
    
    Format your response using markdown for better readability.
    When citing web sources, use the format: [Source Title](URL)
    """
    
    try:
        # Use asyncio.run() to handle the async call in sync context
        response = asyncio.run(get_ai_response(provider, prompt))
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
    # Only return providers that have API keys configured
    available_providers = {
        provider_id: {
            "name": info["name"],
            "enabled": info["enabled"]
        }
        for provider_id, info in AI_PROVIDERS.items()
        if info["api_key"]  # Only include if API key is present
    }
    
    print("Available providers:", available_providers)  # Debug logging
    
    return jsonify({
        "providers": [
            {
                "id": provider_id,
                "name": provider_info["name"],
                "enabled": provider_info["enabled"]
            }
            for provider_id, provider_info in available_providers.items()
        ]
    })

def create_upload_structure(upload_id):
    """Create directory structure for a new upload"""
    base_path = Path(app.config['STORAGE_BASE']) / upload_id
    pdf_path = base_path / 'pdf'
    figures_path = base_path / 'figures'
    chats_path = base_path / 'chats'
    
    # Create directories
    pdf_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)
    chats_path.mkdir(parents=True, exist_ok=True)
    
    return {
        'base_path': base_path,
        'pdf_path': pdf_path,
        'figures_path': figures_path,
        'chats_path': chats_path
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

async def get_ai_response(provider, prompt, temperature=0.7):
    """Generate response from selected AI provider"""
    print(f"Getting AI response from provider: {provider}")
    print(f"Provider enabled status: {AI_PROVIDERS[provider]['enabled']}")
    
    try:
        # Extract the actual content to search for
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
        
        if provider == 'gemini' and AI_PROVIDERS['gemini']['enabled']:
            try:
                print("Generating Gemini response...")
                response = gemini_model.generate_content(
                    prompt,
                    generation_config={"temperature": temperature}
                )
                if not response.text:
                    return "Response blocked due to content safety filters."
                return response.text
            except Exception as e:
                print(f"Gemini error details: {str(e)}")
                raise Exception(f"Gemini error: {str(e)}")
            
        elif provider == 'openai' and AI_PROVIDERS['openai']['disabled']:
            try:
                print("Generating OpenAI response...")
                response = await openai_client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[
                        {"role": "system", "content": "You are a helpful research assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"OpenAI error details: {str(e)}")
                raise Exception(f"OpenAI error: {str(e)}")
            
        elif provider == 'deepseek':
            return "DeepSeek integration coming soon. Please select another AI provider."
            
        else:
            available_providers = [p for p in AI_PROVIDERS if AI_PROVIDERS[p]['enabled']]
            raise ValueError(f"AI provider '{provider}' not available. Available providers: {available_providers}")
            
    except Exception as e:
        print(f"AI response error: {str(e)}")
        raise Exception(str(e))

# After configuring AI_PROVIDERS, add detailed debug logging:
print("\nInitialized AI Providers:")
for provider_id, info in AI_PROVIDERS.items():
    print(f"{provider_id}: enabled = {info['enabled']}, api_key = {'set' if info['api_key'] else 'not set'}")

if __name__ == '__main__':
    print("Starting Flask server in debug mode...")
    app.run(debug=True, port=5000)