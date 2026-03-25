"""
Atlas-OCR (Human-in-the-Loop Refined v1.2)
=========================================
Manual Bounding Box Mode: Target only specific regions for maximum precision.
Engine: PaddleOCR / EasyOCR on high-res crops.
Workflow: Upload PDF -> Draw Bounding Boxes on each page -> Harvest Dimensions.
"""

import os
import re
import cv2
import json
import numpy as np
import pandas as pd
import fitz
import tempfile
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

# Bypassing slow connectivity check and log suppression
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

# Optional: Try to import PaddleOCR or EasyOCR
try:
    from paddleocr import PaddleOCR
    # Note: version 3.4.0+ handles arguments differently; using empty constructor for compatibility.
    _ocr = PaddleOCR()
except ImportError:
    try:
        import easyocr
        # Use GPU=False by default to avoid memory issues for others
        _ocr = easyocr.Reader(['en'], gpu=False)
    except ImportError:
        _ocr = None
        logging.error("No OCR engine (PaddleOCR or EasyOCR) found.")

RENDER_DPI = 300 # Lower for UI responsiveness, can increase for internal cropping
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("atlas-hitl")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024 # 64MB PDF support
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Persistent storage for session (simulated with global for single-user dev)
all_dimensions_harvested = []

# =========================
# Shared Helper: Regex & Normalization
# =========================

def normalize_text(text):
    if not text: return ""
    text = text.replace('–', '-').replace('—', '-') 
    text = text.replace('⌀', 'Ø').replace('ø', 'Ø').replace('Φ', 'Ø').replace('φ', 'Ø').replace('Ö', 'Ø')
    # Handle '0 ' or '0' prefix for diameter
    text = re.sub(r'^[0O]\s*([1-9]\d+)', r'Ø\1', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def parse_dimensions(text):
    results = []
    if not text: return results
    norm = normalize_text(text)
    
    # Threads (M8 X 1.0)
    for m in re.finditer(r'[MH][\s]*(\d+\.?\d*)\s*[xX×\s]*\s*(\d+\.?\d*)?', norm, re.IGNORECASE):
        v = m.group(1)
        if m.group(2): v += f'x{m.group(2)}'
        results.append((v, 'mm', 'thread', f'thread_M{v}'))

    # Holes (6 Holes - Ø6.5)
    for m in re.finditer(r'(\d+)\s*HOLES?[\s.\-/]*Ø?\s*(\d+\.?\d*)', norm, re.IGNORECASE):
        results.append((m.group(1), 'count', 'holes', f'n_holes_{m.group(1)}'))
        results.append((m.group(2), 'mm', 'diameter', f'hole_dia_{m.group(2)}'))

    # Counterbores (CB 10 5)
    for m in re.finditer(r'C\.?B\.?\s*[Ø$]?\s*(\d+\.?\d*)\s*(?:[↓⬇VvL]|OL|OI|dp|depth)?\s*(\d+\.?\d*)?', norm, re.IGNORECASE):
        results.append((m.group(1), 'mm', 'counterbore_dia', f'cb_dia_{m.group(1)}'))
        if m.group(2):
            results.append((m.group(2), 'mm', 'counterbore_depth', f'cb_depth_{m.group(2)}'))

    # Diameters/Radii
    for m in re.finditer(r'Ø\s*(\d+\.?\d*)', norm):
         results.append((m.group(1), 'mm', 'diameter', f'dia_{m.group(1)}'))
    for m in re.finditer(r'R\s*(\d+\.?\d*)', norm, re.IGNORECASE):
        if not re.search(r'[A-Za-z]{3,}', norm):
            results.append((m.group(1), 'mm', 'radius', f'radius_{m.group(1)}'))

    # Linear Fallback
    for m in re.finditer(r'(?<!\d|Ø|M|R|H|[xX×]|B|S)(\d{1,4}\.?\d*)(?!\d)', norm, re.IGNORECASE):
        v = m.group(1)
        if v not in ['0', '00'] and (len(v) >= 2 or '.' in v):
            results.append((v, 'mm', 'linear', f'dim_{v}'))

    return results

# =========================
# PDF Rendering
# =========================

def render_pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    output_files = []
    for i, page in enumerate(doc):
        # We use 300 DPI for standard display and crops
        pix = page.get_pixmap(dpi=300)
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
        
        filename = f"page_{i}_{os.path.basename(pdf_path)}.png"
        path = os.path.join(tempfile.gettempdir(), filename)
        cv2.imwrite(path, bgr)
        output_files.append({"path": path, "name": filename, "width": pix.width, "height": pix.height})
    doc.close()
    return output_files

# =========================
# Routes
# =========================

@app.route('/')
def index():
    # Render with the HITL UI
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({'error': 'No file uploaded'}), 400
    
    # Reset harvest
    global all_dimensions_harvested
    all_dimensions_harvested = []

    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(temp_path)
    
    try:
        pages = render_pdf_to_images(temp_path)
        return jsonify({
            'message': 'File uploaded and rendered.',
            'pages': [{'id': i, 'name': p['name'], 'w': p['width'], 'h': p['height']} for i, p in enumerate(pages)]
        })
    except Exception as e:
        log.exception("Upload fail")
        return jsonify({'error': str(e)}), 500

@app.route('/get_page_image/<filename>')
def get_page_image(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if os.path.exists(path):
        return send_file(path, mimetype='image/png')
    return "Not found", 404

@app.route('/process_crop', methods=['POST'])
def process_crop():
    """
    Receives bounding box coordinates, crops the image and runs OCR.
    """
    data = request.json
    filename = data.get('filename')
    boxes = data.get('boxes', []) # List of {x, y, w, h} (absolute pixels on 300dpi image)
    page_idx = data.get('page_idx', 1)
    
    if not _ocr:
        return jsonify({'error': 'OCR engine not initialized.'}), 500

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if not os.path.exists(image_path):
        return jsonify({'error': 'Page image not found.'}), 404

    img = cv2.imread(image_path)
    page_results = []

    for i, box in enumerate(boxes):
        log.info(f"Processing box {i+1} for page {page_idx+1}: {box}")
        x, y, w, h = map(int, [box['x'], box['y'], box['w'], box['h']])
        # Buffer the crop slightly to prevent clipped text
        x_buf, y_buf = max(0, x-10), max(0, y-10)
        # Handle image boundaries
        h_img, w_img = img.shape[:2]
        crop = img[y_buf : min(h_img, y+h+10), x_buf : min(w_img, x+w+10)]
        
        if crop.size == 0: 
            log.warning(f"Crop {i+1} is empty.")
            continue

        text = ""
        try:
            # For manual boxes, we skip detection (det=False) and just do Recognition
            if hasattr(_ocr, 'ocr'): # PaddleOCR
                log.info(f"Running PaddleOCR on crop {i+1}...")
                ocr_res = _ocr.ocr(crop, det=False, cls=True)
                log.info(f"Raw OCR Output: {ocr_res}")
                
                if ocr_res:
                    # v5 and PADDLEX integration might return results differently
                    # Standard check for [[text, conf], ...]
                    if isinstance(ocr_res, list) and len(ocr_res) > 0:
                        inner = ocr_res[0]
                        if isinstance(inner, list) and len(inner) > 0:
                            # Typical: [[text, conf], ...]
                            text = " ".join([str(line[0]) for line in inner if isinstance(line, (list, tuple))])
                        elif isinstance(inner, (list, tuple)):
                             # Fallback: [text, conf]
                             text = str(inner[0])
            else: # EasyOCR
                ocr_res = _ocr.readtext(crop)
                if ocr_res:
                    text = " ".join([res[1] for res in ocr_res])
        except Exception as e:
            log.error(f"OCR error on box {i+1}: {e}")
            continue

        log.info(f"Box {i+1} Final Text: '{text}'")
        if not text:
             log.warning(f"No text extracted for box {i+1}")
             continue

        found = parse_dimensions(text)
        log.info(f"Box {i+1} Parsed Results: {found}")
        for (v, u, t, f) in found:
            entry = {
                'id': len(all_dimensions_harvested) + 1,
                'page': page_idx + 1,
                'feature': f,
                'value': v,
                'type': t,
                'unit': u,
                'raw_text': text,
                'notes': f"Manual Box {page_idx+1}"
            }
            all_dimensions_harvested.append(entry)
            page_results.append(entry)

    return jsonify({'results': page_results})

@app.route('/export', methods=['GET'])
def export():
    if not all_dimensions_harvested:
        return jsonify({'error': 'No harvested dimensions to export.'}), 400
    
    df = pd.DataFrame(all_dimensions_harvested)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'harvested_dimensions_{ts}.xlsx'
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Reorder columns
    cols = ['id', 'page', 'feature', 'value', 'type', 'unit', 'raw_text', 'notes']
    existing = [c for c in cols if c in df.columns]
    df[existing].to_excel(path, index=False)
    
    return send_file(path, as_attachment=True, download_name=filename)

if __name__ == '__main__':
    # Force single threading to avoid OCR engine crashes
    app.run(debug=True, port=5000, threaded=False)