"""
Atlas-OCR v1 — PaddleOCR Backend
==================================
* Same API as app.py  →  works with the same index.html frontend
* Uses PaddleOCR (CPU, no GPU required)
* Improved parse_dimensions with span-tracking (no duplicate extraction)
* Run: python app-v1.py
"""

import os
import re
import cv2
import base64
import numpy as np
import pandas as pd
import fitz
import tempfile
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

# ── suppress noisy paddle logs ──
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['FLAGS_logtostderr'] = '0'
os.environ['FLAGS_minloglevel'] = '3'

# ── PaddleOCR init (lazy) ──
_ocr = None

def get_ocr():
    global _ocr
    if _ocr is None:
        try:
            from paddleocr import PaddleOCR
            log.info("Loading PaddleOCR...")
            _ocr = PaddleOCR(lang='en', show_log=False, use_angle_cls=True)
            log.info("PaddleOCR ready.")
        except Exception as e:
            log.error(f"PaddleOCR failed to load: {e}")
            _ocr = "FAILED"
    return _ocr

# ── Config ──
RENDER_DPI        = 300
MIN_CONFIDENCE    = 0.5
ALLOWED_EXT       = {'pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("atlas-paddle")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()


# ════════════════════════════════════════════════════════
# TEXT NORMALIZATION & DIMENSION PARSING
# ════════════════════════════════════════════════════════

def normalize_text(text):
    if not text:
        return ""
    text = text.strip()
    text = text.replace('–', '-').replace('—', '-')
    text = text.replace('⌀', 'Ø').replace('ø', 'Ø')
    text = text.replace('Φ', 'Ø').replace('φ', 'Ø')
    text = text.replace('Ö', 'Ø').replace('ö', 'Ø')
    # Mis-read diameter prefix: $12, O12, 0133
    text = re.sub(r'\$\s*(\d)', r'Ø\1', text)
    text = re.sub(r'^[0O]\s*([1-9]\d+)', r'Ø\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def parse_dimensions(text):
    """
    Extract engineering dimensions from OCR text.
    Returns list of (value, unit, type, feature_name, confidence) tuples.
    Patterns are checked in priority order; consumed spans are excluded
    from the linear fallback to prevent over-extraction.
    """
    results = []
    consumed_spans = set()
    if not text:
        return results

    norm = normalize_text(text)

    def mark(m):
        for i in range(m.start(), m.end()):
            consumed_spans.add(i)

    try:
        # ── Priority 1: Threads  M8  M10x1.5 ──
        for m in re.finditer(r'\bM(\d+\.?\d*)(?:\s*[xX×]\s*(\d+\.?\d*))?', norm, re.IGNORECASE):
            v = m.group(1)
            if m.group(2):
                v += f'x{m.group(2)}'
            results.append((v, 'mm', 'thread', f'thread_M{v}', 0.92))
            mark(m)

        # ── Priority 2: Holes  6 HOLES Ø6.5 ──
        for m in re.finditer(r'(\d+)\s*HOLES?\s*[-–—\s]*[Ø$]?\s*(\d+\.?\d*)', norm, re.IGNORECASE):
            results.append((m.group(1), 'count', 'hole_count', f'holes_{m.group(1)}', 0.92))
            results.append((m.group(2), 'mm', 'hole_diameter', f'hole_dia_{m.group(2)}', 0.90))
            mark(m)

        # ── Priority 3: Counterbore  CB Ø10 ↓5 ──
        for m in re.finditer(
            r'C\.?B\.?\s*[Ø$]?\s*(\d+\.?\d*)(?:\s*(?:[↓⬇VvL]|depth)\s*(\d+\.?\d*))?',
            norm, re.IGNORECASE
        ):
            results.append((m.group(1), 'mm', 'counterbore_dia', f'cb_dia_{m.group(1)}', 0.88))
            if m.group(2):
                results.append((m.group(2), 'mm', 'counterbore_depth', f'cb_depth_{m.group(2)}', 0.85))
            mark(m)

        # ── Priority 4: Diameter  Ø12.5 ──
        for m in re.finditer(r'Ø\s*(\d+\.?\d*)', norm):
            results.append((m.group(1), 'mm', 'diameter', f'dia_{m.group(1)}', 0.90))
            mark(m)

        # ── Priority 5: Radius  R10 ──
        for m in re.finditer(r'\bR(\d+\.?\d*)\b', norm):
            before = norm[max(0, m.start() - 1):m.start()]
            if not before or not before[-1].isalpha():
                results.append((m.group(1), 'mm', 'radius', f'radius_{m.group(1)}', 0.87))
                mark(m)

        # ── Priority 6: Linear fallback — only numbers NOT already matched ──
        already_found_values = {r[0] for r in results}
        for m in re.finditer(r'(?<![A-Za-zØ.])(\d+\.\d+|\d{2,})(?![A-Za-z.\d])', norm):
            if any(i in consumed_spans for i in range(m.start(), m.end())):
                continue
            v = m.group(1)
            # Skip if this is a leading-zero misread of an already-found value
            # e.g. '0169' when 'Ø169' → '169' already captured as diameter
            v_stripped = v.lstrip('0') or '0'
            if v_stripped in already_found_values:
                continue
            if v in ('00', '000') or float(v) > 9999:
                continue
            results.append((v, 'mm', 'linear', f'dim_{v}', 0.80))


    except Exception as e:
        log.error(f"parse_dimensions error: {e}")

    return results


def deduplicate(dims):
    """Keep highest-confidence entry per (value, type) pair."""
    seen = {}
    for d in dims:
        key = (d[0], d[2])
        if key not in seen or d[4] > seen[key][4]:
            seen[key] = d
    return list(seen.values())


# ════════════════════════════════════════════════════════
# PDF → IMAGE RENDERING
# ════════════════════════════════════════════════════════

def pdf_to_images(pdf_path):
    """Render each PDF page to a PNG at RENDER_DPI, return list of file paths."""
    doc = fitz.open(pdf_path)
    out = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=RENDER_DPI)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
        p = os.path.join(tempfile.gettempdir(), f'page_{i}_{os.path.basename(pdf_path)}.png')
        cv2.imwrite(p, bgr)
        out.append(p)
    doc.close()
    return out


def encode_b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


# ════════════════════════════════════════════════════════
# OCR CROP & PARSE
# ════════════════════════════════════════════════════════

def ocr_text_from_crop(crop_bgr):
    """Run PaddleOCR on a crop and return the extracted text string."""
    ocr = get_ocr()
    if not ocr or ocr == "FAILED":
        return ""

    try:
        res = ocr.ocr(crop_bgr)
        if not res:
            return ""
        lines = []
        for page_res in res:
            if not page_res:
                continue
            for item in page_res:
                # item = [bbox, (text, conf)]  OR  [text, conf]  (older paddle)
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    text_part = item[1]
                    if isinstance(text_part, (list, tuple)):
                        t, c = text_part[0], text_part[1]
                    else:
                        t, c = str(text_part), 1.0
                    if c >= MIN_CONFIDENCE and t.strip():
                        lines.append(t.strip())
        return " ".join(lines)
    except Exception as e:
        log.error(f"PaddleOCR crop error: {e}")
        return ""


def crop_and_ocr_box(image_bgr, box_norm, page_num, box_idx):
    """Crop using normalized coords, run OCR, return list of dimension tuples."""
    h, w = image_bgr.shape[:2]
    buf = 15
    x1 = max(0, int(box_norm['x1'] * w) - buf)
    y1 = max(0, int(box_norm['y1'] * h) - buf)
    x2 = min(w, int(box_norm['x2'] * w) + buf)
    y2 = min(h, int(box_norm['y2'] * h) + buf)

    if x2 <= x1 or y2 <= y1:
        log.warning(f"  Box {box_idx}: Invalid region.")
        return []

    crop = image_bgr[y1:y2, x1:x2]
    log.info(f"  Box {box_idx}: crop ({x1},{y1})->({x2},{y2}) on ({w}x{h})")

    raw_text = ocr_text_from_crop(crop)
    log.info(f"  Box {box_idx}: OCR text = '{raw_text}'")

    if not raw_text.strip():
        log.warning(f"  Box {box_idx}: No text detected.")
        return []

    # Single pass on combined text
    found = parse_dimensions(raw_text)

    # Word-level fallback for values missed in combined parse
    captured = {d[0] for d in found}
    for word in raw_text.split():
        word_dims = parse_dimensions(word)
        for wd in word_dims:
            if wd[0] not in captured:
                found.append(wd)
                captured.add(wd[0])

    return found


# ════════════════════════════════════════════════════════
# MAIN PROCESSING
# ════════════════════════════════════════════════════════

def process_drawing_with_annotations(filepath, filename, annotations):
    """
    Crop each annotated box from the high-DPI image, run PaddleOCR,
    parse dimensions, deduplicate, and export.
    """
    cleanup = []
    try:
        ext = filename.lower().rsplit('.', 1)[-1]
        if ext == 'pdf':
            image_paths = pdf_to_images(filepath)
            cleanup.extend(image_paths)
        else:
            image_paths = [filepath]

        all_data = []
        seen = set()

        for page_idx, image_path in enumerate(image_paths):
            page_num = page_idx + 1
            page_annos = annotations.get(str(page_num), [])
            if not page_annos:
                continue

            log.info(f"Page {page_num}: {len(page_annos)} box(es)")
            img = cv2.imread(image_path)
            if img is None:
                log.error(f"Page {page_num}: Cannot read {image_path}")
                continue

            for box_idx, box in enumerate(page_annos):
                raw_dims = crop_and_ocr_box(img, box, page_num, box_idx + 1)
                unique = deduplicate(raw_dims)
                for (v, u, t, f, conf) in unique:
                    key = (page_num, v, t)
                    if key not in seen:
                        seen.add(key)
                        all_data.append({
                            'id': str(len(all_data) + 1),
                            'feature': f,
                            'value': v,
                            'unit': u,
                            'type': t,
                            'confidence': round(conf, 2),
                            'notes': f'page {page_num}, box {box_idx + 1}'
                        })

        log.info(f"Total extracted: {len(all_data)}")
        if not all_data:
            return None, "No dimensions found. Draw larger boxes around text.", None, None

        df = pd.DataFrame(all_data)
        cols = ['id', 'feature', 'value', 'unit', 'type', 'confidence', 'notes']
        df = df[cols]

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        xf = f'dims_paddle_{ts}.xlsx'
        cf = f'dims_paddle_{ts}.csv'
        df.to_excel(os.path.join(tempfile.gettempdir(), xf), index=False)
        df.to_csv(os.path.join(tempfile.gettempdir(), cf), index=False)

        return {
            'columns': cols,
            'data': all_data,
            'total_dimensions': len(all_data),
            'total_pages': len(image_paths)
        }, None, xf, cf

    except Exception as e:
        log.error(f"Processing error: {e}", exc_info=True)
        return None, str(e), None, None
    finally:
        for p in cleanup:
            try: os.remove(p)
            except: pass


# ════════════════════════════════════════════════════════
# FLASK ROUTES  ←  identical API to app.py
# ════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        if not file.filename:
            return jsonify({'error': 'Empty filename'}), 400

        ext = file.filename.lower().rsplit('.', 1)[-1] if '.' in file.filename else ''
        if ext not in ALLOWED_EXT:
            return jsonify({'error': f'Invalid type. Allowed: {", ".join(ALLOWED_EXT)}'}), 400

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(file.filename)
        filepath = os.path.join(tempfile.gettempdir(), f'{ts}_{filename}')
        file.save(filepath)
        log.info(f"Uploaded: {filepath}")

        if ext == 'pdf':
            image_paths = pdf_to_images(filepath)
        else:
            image_paths = [filepath]

        images_b64 = [encode_b64(p) for p in image_paths]

        return jsonify({
            'success': True,
            'filepath': filepath,
            'filename': filename,
            'total_pages': len(image_paths),
            'images': images_b64,
            'image_paths': image_paths
        })
    except Exception as e:
        log.error(f"Upload error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/process', methods=['POST'])
def process():
    try:
        data = request.get_json()
        if not data or 'filepath' not in data or 'annotations' not in data:
            return jsonify({'error': 'Missing filepath or annotations'}), 400

        filepath    = data['filepath']
        annotations = data['annotations']
        filename    = data.get('filename', 'document')

        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found on server'}), 404

        results, error, xf, cf = process_drawing_with_annotations(filepath, filename, annotations)
        if error:
            return jsonify({'error': error}), 500

        return jsonify({'success': True, 'results': results, 'xlsx_file': xf, 'csv_file': cf})

    except Exception as e:
        log.error(f"Process error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/<filename>')
def download(filename):
    try:
        path = os.path.join(tempfile.gettempdir(), secure_filename(filename))
        if not os.path.exists(path):
            return jsonify({'error': 'File not found'}), 404
        return send_file(path, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health')
def health():
    return jsonify({'status': 'paddle-ocr-backend', 'timestamp': datetime.now().isoformat()})


if __name__ == '__main__':
    log.info("Starting Atlas-OCR (PaddleOCR backend) on port 5001...")
    app.run(debug=True, port=5001, threaded=False)