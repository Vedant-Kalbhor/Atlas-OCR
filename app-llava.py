from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import pandas as pd
import tempfile
from datetime import datetime
from werkzeug.utils import secure_filename
import re

import fitz  # PyMuPDF
import ollama

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'bmp'}

# ------------------ Helpers ------------------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_json(text):
    """
    Extract the first JSON object found in text.
    Handles extra text, markdown, explanations.
    """
    match = re.search(r'\{[\s\S]*\}', text)
    if not match:
        return None
    return match.group(0)


def pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    images = []

    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=300)
        img_path = f"{pdf_path}_page_{i}.png"
        pix.save(img_path)
        images.append(img_path)

    return images


def call_llava(image_paths, strict=False):
    """
    Calls LLaVA with a strict JSON-only prompt.
    """

    prompt = """
You are a data extraction engine.

TASK:
Extract ALL dimensions from the engineering drawing.

OUTPUT RULES (MANDATORY):
- Output MUST be valid JSON
- Output MUST start with '{' and end with '}'
- NO text before or after JSON
- NO markdown
- NO explanations
- Use "N/A" for missing values

JSON FORMAT:
{
  "columns": ["id", "..."],
  "data": [
    {"id": "1", "...": "..."}
  ]
}

UNITS:
- Include units in column names (mm, inch, cm)

FAILURE CONDITION:
If you cannot comply, still output EXACTLY:
{
  "columns": ["id"],
  "data": []
}
"""

    messages = [{
        "role": "user",
        "content": prompt,
        "images": image_paths
    }]

    response = ollama.chat(
        model="llava:latest",
        messages=messages
    )

    return response["message"]["content"]


# ------------------ Routes ------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    temp_dir = tempfile.gettempdir()
    filename = secure_filename(file.filename)
    filepath = os.path.join(temp_dir, filename)

    image_paths = []

    try:
        file.save(filepath)

        if filename.lower().endswith('.pdf'):
            image_paths = pdf_to_images(filepath)
            os.remove(filepath)
        else:
            image_paths = [filepath]

        # ---------- LLaVA with retry ----------
        data = None
        last_response = None

        for attempt in range(2):
            llava_response = call_llava(image_paths)
            last_response = llava_response

            raw = llava_response.strip()

            # Remove markdown if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                raw = raw.replace("json", "").strip()

            json_text = extract_json(raw)

            if json_text:
                try:
                    data = json.loads(json_text)
                    break
                except json.JSONDecodeError:
                    pass  # retry

        if data is None:
            return jsonify({
                "error": "Model failed to produce valid JSON",
                "raw_response": last_response
            }), 500

        # ---------- Excel ----------
        excel_name = f"dimensions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        excel_path = os.path.join(temp_dir, excel_name)

        df = pd.DataFrame(data.get("data", []))
        df.to_excel(excel_path, index=False, engine="openpyxl")

        return jsonify({
            "preview": data,
            "excel_file": excel_name
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # Cleanup images
        for img in image_paths:
            if os.path.exists(img):
                os.remove(img)


@app.route('/download/<filename>')
def download(filename):
    temp_dir = tempfile.gettempdir()
    path = os.path.join(temp_dir, secure_filename(filename))

    if not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404

    return send_file(
        path,
        as_attachment=True,
        download_name=filename,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )


if __name__ == '__main__':
    app.run(debug=True, port=5000)
