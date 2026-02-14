"""
Atlas-OCR: Local Engineering Drawing Dimension Extractor
=========================================================
Hybrid approach: EasyOCR + Tesseract OCR + OpenCV geometry detection + LLaVA vision model (Ollama).
All processing is done locally — no API keys, no cloud services.

Pipeline:
  1. PDF/Image → high-DPI rasterization (PyMuPDF)
  2. Dual-engine OCR (EasyOCR + Tesseract) for comprehensive text extraction
  3. OpenCV geometry detection (circles, lines, arrows)
  4. LLaVA vision model for visual understanding + structured JSON output
  5. Post-processing, deduplication, and export to XLSX/CSV
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import re
import cv2
import numpy as np
import pandas as pd
import tempfile
import logging
from datetime import datetime
from werkzeug.utils import secure_filename

import fitz  # PyMuPDF
import easyocr
import pytesseract
import ollama

# ──────────────────── CONFIG ────────────────────

# Tesseract path (Windows default)
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Ollama vision model
VISION_MODEL = "llava:latest"

# Rendering DPI for PDF pages
RENDER_DPI = 300

# Max image dimension sent to LLaVA (to keep memory manageable)
MAX_LLAVA_DIM = 1536

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}

# ──────────────────── LOGGING ────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("atlas-ocr")

# ──────────────────── FLASK ────────────────────

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB

# ──────────────────── LAZY GLOBALS ────────────────────

_easyocr_reader = None


def get_easyocr_reader():
    """Lazy-init EasyOCR reader (slow first load)."""
    global _easyocr_reader
    if _easyocr_reader is None:
        log.info("Initializing EasyOCR reader (first request may be slow)...")
        _easyocr_reader = easyocr.Reader(['en'], gpu=False)
        log.info("EasyOCR reader ready.")
    return _easyocr_reader


# ──────────────────── UTILITIES ────────────────────

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def pdf_to_images(pdf_path, dpi=RENDER_DPI):
    """Convert each page of a PDF to a high-res BGR numpy array."""
    doc = fitz.open(pdf_path)
    images = []
    img_paths = []

    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Save to disk for LLaVA
        img_path = pdf_path + f"_page_{i}.png"
        cv2.imwrite(img_path, img_bgr)
        images.append(img_bgr)
        img_paths.append(img_path)

    doc.close()
    return images, img_paths


def resize_for_llava(img_path, max_dim=MAX_LLAVA_DIM):
    """Resize image if too large, return path to resized image."""
    img = cv2.imread(img_path)
    if img is None:
        return img_path
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img_path
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized_path = img_path.replace(".png", "_resized.png")
    cv2.imwrite(resized_path, resized)
    return resized_path


# ──────────────────── OCR PIPELINE ────────────────────

def run_easyocr(img_bgr, page_no):
    """Run EasyOCR on the image (single pass for speed)."""
    reader = get_easyocr_reader()
    results = []

    try:
        ocr_results = reader.readtext(img_bgr)
        for bbox, text, confidence in ocr_results:
            text = text.strip()
            if not text:
                continue
            x_coords = [pt[0] for pt in bbox]
            y_coords = [pt[1] for pt in bbox]
            cx = int(sum(x_coords) / len(x_coords))
            cy = int(sum(y_coords) / len(y_coords))

            results.append({
                'text': text,
                'confidence': float(confidence),
                'cx': cx,
                'cy': cy,
                'source': 'easyocr',
                'page': page_no
            })
    except Exception as e:
        log.warning(f"EasyOCR failed on page {page_no}: {e}")

    return results


def run_tesseract(img_bgr, page_no):
    """Run Tesseract OCR on the image."""
    results = []
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    try:
        data = pytesseract.image_to_data(binary, config="--psm 6", output_type=pytesseract.Output.DICT)
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            if not text or conf < 30:
                continue

            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]

            results.append({
                'text': text,
                'confidence': conf / 100.0,
                'cx': x + w // 2,
                'cy': y + h // 2,
                'source': 'tesseract',
                'page': page_no
            })
    except Exception as e:
        log.warning(f"Tesseract failed on page {page_no}: {e}")

    return results


def merge_ocr_results(easyocr_results, tesseract_results):
    """Merge and deduplicate results from both OCR engines."""
    all_results = easyocr_results + tesseract_results
    if not all_results:
        return []

    all_results.sort(key=lambda x: x['confidence'], reverse=True)

    merged = []
    used_positions = []

    for r in all_results:
        is_duplicate = False
        for existing in used_positions:
            if (abs(r['cx'] - existing['cx']) < 30 and
                abs(r['cy'] - existing['cy']) < 30 and
                r['text'] == existing['text']):
                is_duplicate = True
                break
        if not is_duplicate:
            merged.append(r)
            used_positions.append(r)

    return merged


# ──────────────────── GEOMETRY DETECTION ────────────────────

def detect_geometry(img_bgr):
    """Detect circles and dimension lines/arrows in the drawing."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    circles_raw = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
        param1=120, param2=30, minRadius=10, maxRadius=400
    )
    circles = []
    if circles_raw is not None:
        for c in np.uint16(np.around(circles_raw[0])):
            circles.append({'x': int(c[0]), 'y': int(c[1]), 'r': int(c[2])})

    edges = cv2.Canny(gray, 50, 150)
    lines_raw = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=30, maxLineGap=10)
    arrows = []
    if lines_raw is not None:
        for l in lines_raw:
            x1, y1, x2, y2 = l[0]
            length = np.hypot(x2 - x1, y2 - y1)
            if length > 40:
                arrows.append({'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)})

    return circles, arrows


# ──────────────────── OCR TEXT CLEANING ────────────────────

def clean_ocr_text(text):
    """Clean common OCR artifacts in dimension text."""
    # Remove spaces within numbers: '41 0' -> '410', '1 0' -> '10'
    cleaned = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    # Normalize diameter symbols
    cleaned = cleaned.replace('⌀', 'Ø').replace('ø', 'Ø')
    return cleaned.strip()


# ──────────────────── OCR CONTEXT BUILDER ────────────────────

def build_ocr_context(ocr_results):
    """Build a compact textual summary of OCR results for the LLM prompt."""
    dim_pattern = re.compile(r'(\d+\.?\d*)', re.IGNORECASE)
    dimension_texts = []
    label_texts = []

    for r in ocr_results:
        # Clean the text
        r_copy = dict(r)
        r_copy['text'] = clean_ocr_text(r_copy['text'])
        if dim_pattern.search(r_copy['text']):
            dimension_texts.append(r_copy)
        else:
            label_texts.append(r_copy)

    return dimension_texts, label_texts


# ──────────────────── LLAVA VISION CALL ────────────────────

def call_llava_with_context(image_paths, ocr_results, retry_count=3):
    """
    Call LLaVA vision model with OCR-detected numbers as primary data.
    LLaVA's job is to CLASSIFY and LABEL the numbers, not discover new ones.
    """

    dimension_texts, label_texts = ocr_results

    # Build the list of OCR-detected numbers for LLaVA to classify
    ocr_numbers = []
    for i, r in enumerate(dimension_texts, start=1):
        ocr_numbers.append(f"  {i}. \"{r['text']}\"")

    ocr_labels = []
    for r in label_texts[:15]:
        ocr_labels.append(f"  - \"{r['text']}\"")

    numbers_list = "\n".join(ocr_numbers) if ocr_numbers else "  (none detected)"
    labels_list = "\n".join(ocr_labels) if ocr_labels else "  (none detected)"

    prompt = f"""Analyze this engineering drawing. I need you to identify what each detected number/dimension represents.

These numbers were found by OCR in the drawing:
{numbers_list}

Text labels found:
{labels_list}

For EACH number above, tell me:
- "feature": a short name like outer_diameter, total_length, slot_width, bore_depth, fillet_radius etc.
- "value": copy the exact number from the OCR list
- "unit": write mm
- "type": one word from: linear, diameter, radius, angle, depth, width, height, chamfer, thread
- "notes": which part or view it belongs to

Rules:
- Include ALL the OCR numbers listed above, do not skip any
- Use the EXACT numbers from the OCR list as values
- Give each dimension a descriptive feature name based on what you see in the drawing
- Do NOT add any numbers that are not in the OCR list

Output ONLY valid JSON with this structure, nothing else:
{{"columns":["id","feature","value","unit","type","notes"],"data":[...]}}"""

    resized_paths = [resize_for_llava(p) for p in image_paths]

    last_raw = None
    for attempt in range(retry_count):
        try:
            log.info(f"  LLaVA attempt {attempt + 1}/{retry_count}...")
            response = ollama.chat(
                model=VISION_MODEL,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": resized_paths
                }]
            )

            raw = response["message"]["content"].strip()
            last_raw = raw
            log.info(f"  LLaVA response length: {len(raw)} chars")
            log.info(f"  LLaVA raw: {raw[:500]}")

            parsed = extract_json_from_response(raw)
            if parsed is not None and 'data' in parsed and len(parsed['data']) > 0:
                return parsed, raw

            log.warning(f"  Attempt {attempt + 1}: JSON parse failed or empty data")

        except Exception as e:
            log.error(f"  LLaVA attempt {attempt + 1} error: {e}")
            last_raw = str(e)

    return None, last_raw


def extract_json_from_response(text):
    """Extract and parse JSON from LLM response."""
    # Remove markdown code blocks
    if '```' in text:
        parts = text.split('```')
        for part in parts:
            cleaned = part.strip()
            if cleaned.startswith('json'):
                cleaned = cleaned[4:].strip()
            if cleaned.startswith('{'):
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    pass

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Regex extraction
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ──────────────────── OCR-ONLY FALLBACK ────────────────────

def ocr_only_extraction(ocr_results, circles, arrows, page_no):
    """Fallback: extract dimensions using only OCR + geometry (no LLM)."""
    dims = []
    idx = 1
    used = set()

    for w in ocr_results:
        text = w['text'].replace('⌀', '').replace('Ø', '').replace('ø', '').strip()
        if not any(ch.isdigit() for ch in text):
            continue

        num_match = re.search(r'(\d+\.?\d*)', text)
        if not num_match:
            continue

        value = num_match.group(1)
        dim_type = 'linear'

        # Check proximity to circles → diameter
        for c in circles:
            dist = np.hypot(float(w['cx']) - c['x'], float(w['cy']) - c['y'])
            if dist < c['r'] * 1.8:
                dim_type = 'diameter'
                break

        # Radius
        if text.upper().startswith('R'):
            dim_type = 'radius'

        # Angle
        if '°' in text or 'deg' in text.lower():
            dim_type = 'angle'

        key = (page_no, value, dim_type)
        if key in used:
            continue
        used.add(key)

        dims.append({
            'id': str(idx),
            'feature': f'{dim_type}_{idx}',
            'value': value,
            'unit': 'mm',
            'type': dim_type,
            'notes': f'page {page_no}, OCR ({w["source"]})'
        })
        idx += 1

    columns = ['id', 'feature', 'value', 'unit', 'type', 'notes']
    return {'columns': columns, 'data': dims}


# ──────────────────── MAIN PROCESSING ────────────────────

def process_drawing(filepath, filename):
    """Main processing pipeline."""
    temp_dir = tempfile.gettempdir()
    cleanup_paths = []

    try:
        # ─── Step 1: Convert to images ───
        log.info("Step 1: Converting to images...")
        if filename.lower().endswith('.pdf'):
            images, img_paths = pdf_to_images(filepath)
            cleanup_paths.extend(img_paths)
        else:
            img = cv2.imread(filepath)
            if img is None:
                raise ValueError("Could not read image file")
            img_path = os.path.join(temp_dir, filename + "_processed.png")
            cv2.imwrite(img_path, img)
            images = [img]
            img_paths = [img_path]
            cleanup_paths.append(img_path)

        log.info(f"  {len(images)} page(s) to process")

        all_data = []
        standard_columns = ['id', 'feature', 'value', 'unit', 'type', 'notes']

        for page_idx, (img_bgr, img_path) in enumerate(zip(images, img_paths)):
            page_no = page_idx + 1
            log.info(f"Processing page {page_no}/{len(images)}...")

            # ─── Step 2: OCR ───
            log.info("  Running EasyOCR...")
            easyocr_results = run_easyocr(img_bgr, page_no)
            log.info(f"  EasyOCR: {len(easyocr_results)} text regions")

            log.info("  Running Tesseract...")
            tesseract_results = run_tesseract(img_bgr, page_no)
            log.info(f"  Tesseract: {len(tesseract_results)} text regions")

            merged_ocr = merge_ocr_results(easyocr_results, tesseract_results)
            log.info(f"  Merged: {len(merged_ocr)} unique text regions")

            # ─── Step 3: Geometry detection ───
            circles, arrows = detect_geometry(img_bgr)
            log.info(f"  Geometry: {len(circles)} circles, {len(arrows)} lines/arrows")

            # ─── Step 4: Build context ───
            ocr_context = build_ocr_context(merged_ocr)

            # ─── Step 5: LLaVA analysis ───
            log.info("  Calling LLaVA vision model...")
            llava_data, llava_raw = call_llava_with_context([img_path], ocr_context)

            if llava_data and 'data' in llava_data and llava_data['data']:
                log.info(f"  LLaVA extracted {len(llava_data['data'])} dimension rows")
                # Normalize to standard columns where possible
                for row in llava_data['data']:
                    # Add page info to notes
                    if 'notes' in row:
                        if f'page {page_no}' not in str(row.get('notes', '')):
                            row['notes'] = f"page {page_no}, {row['notes']}"
                    else:
                        row['notes'] = f"page {page_no}"
                all_data.extend(llava_data['data'])
                # Track columns from LLaVA
                if 'columns' in llava_data:
                    for col in llava_data['columns']:
                        if col not in standard_columns:
                            standard_columns.append(col)
            else:
                log.info("  Falling back to OCR-only extraction")
                page_data = ocr_only_extraction(merged_ocr, circles, arrows, page_no)
                if page_data and page_data['data']:
                    all_data.extend(page_data['data'])

        # ─── Step 6: Build final result ───
        if not all_data:
            return None, "No dimensions could be extracted from the drawing.", None, None

        # Re-index IDs
        for i, row in enumerate(all_data, start=1):
            row['id'] = str(i)

        # Determine final columns
        all_keys = set()
        for row in all_data:
            all_keys.update(row.keys())

        # Order: standard columns first, then any extra
        final_columns = [c for c in standard_columns if c in all_keys]
        extra_cols = sorted([c for c in all_keys if c not in final_columns])
        final_columns.extend(extra_cols)

        result = {
            'columns': final_columns,
            'data': all_data
        }

        # ─── Step 7: Export ───
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        excel_name = f'dimensions_{ts}.xlsx'
        csv_name = f'dimensions_{ts}.csv'
        excel_path = os.path.join(temp_dir, excel_name)
        csv_path = os.path.join(temp_dir, csv_name)

        df = pd.DataFrame(all_data)
        existing_cols = [c for c in final_columns if c in df.columns]
        df = df[existing_cols]

        df.to_excel(excel_path, index=False, engine='openpyxl')
        df.to_csv(csv_path, index=False)

        log.info(f"Exported {len(all_data)} rows to {excel_name}")

        return result, None, excel_name, csv_name

    finally:
        for p in cleanup_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
                resized = p.replace(".png", "_resized.png")
                if os.path.exists(resized):
                    os.remove(resized)
            except:
                pass


# ──────────────────── FLASK ROUTES ────────────────────

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
        return jsonify({'error': 'Invalid file type. Supported: PDF, PNG, JPG, JPEG, BMP, TIF'}), 400

    temp_dir = tempfile.gettempdir()
    filename = secure_filename(file.filename)
    filepath = os.path.join(temp_dir, filename)

    try:
        file.save(filepath)
        result_data, error, excel_name, csv_name = process_drawing(filepath, filename)

        if error:
            return jsonify({'error': error}), 500

        return jsonify({
            'preview': result_data,
            'excel_file': excel_name,
            'csv_file': csv_name
        })

    except Exception as e:
        log.exception("Error processing file")
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

    finally:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass


@app.route('/download/<filename>')
def download(filename):
    temp_dir = tempfile.gettempdir()
    safe_name = secure_filename(filename)
    path = os.path.join(temp_dir, safe_name)

    if not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404

    mimetype = 'text/csv' if safe_name.endswith('.csv') else \
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'

    return send_file(path, as_attachment=True, download_name=safe_name, mimetype=mimetype)


if __name__ == '__main__':
    log.info("=" * 60)
    log.info("    Atlas-OCR: Local Dimension Extractor")
    log.info("=" * 60)
    log.info(f"  Tesseract : {pytesseract.pytesseract.tesseract_cmd}")
    log.info(f"  Vision LLM: {VISION_MODEL}")
    log.info(f"  Render DPI: {RENDER_DPI}")
    log.info(f"  LLaVA Max : {MAX_LLAVA_DIM}px")
    log.info("=" * 60)
    app.run(debug=True, port=5000, use_reloader=False)
