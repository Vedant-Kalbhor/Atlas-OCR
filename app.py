from flask import Flask, render_template, request, jsonify, send_file
import os
import cv2
import numpy as np
import pandas as pd
import tempfile
from datetime import datetime
from pdf2image import convert_from_path
from werkzeug.utils import secure_filename
from paddleocr import PaddleOCR

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}

# =========================
# PaddleOCR
# =========================
ocr = PaddleOCR(
    lang='en',
    use_textline_orientation=True
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# =========================
# PDF → Images
# =========================
def convert_pdf_to_images(pdf_path, dpi=300):
    pages = convert_from_path(pdf_path, dpi=dpi)
    return [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pages]

# =========================
# Geometry Detection
# =========================
def detect_circles(gray):
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=40,
        param1=120, param2=30,
        minRadius=10, maxRadius=400
    )
    results = []
    if circles is not None:
        for c in np.uint16(np.around(circles[0])):
            results.append({'x': int(c[0]), 'y': int(c[1]), 'r': int(c[2])})
    return results

def detect_dimension_lines(gray):
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180,
        threshold=120,
        minLineLength=80,
        maxLineGap=5
    )
    dim_lines = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            length = np.hypot(x2-x1, y2-y1)
            angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
            if length > 80 and (angle < 10 or abs(angle - 90) < 10):
                dim_lines.append((x1, y1, x2, y2))
    return dim_lines

# =========================
# OCR
# =========================
def run_paddle_ocr(image):
    result = ocr.predict(image)
    words = []

    for res in result:
        boxes = res.get('dt_polys', [])
        texts = res.get('rec_texts', [])
        scores = res.get('rec_scores', [])

        for box, text, score in zip(boxes, texts, scores):
            x_coords = [p[0] for p in box]
            y_coords = [p[1] for p in box]

            words.append({
                'text': text.replace('⌀', '').replace('Ø', ''),
                'confidence': float(score),
                'cx': int(sum(x_coords) / len(x_coords)),
                'cy': int(sum(y_coords) / len(y_coords))
            })
    return words

# =========================
# Utility Filters
# =========================
def is_valid_dimension(text):
    try:
        val = float(text)
        return 0.5 < val < 5000
    except:
        return False

def cluster_dimensions(dims, tol=3):
    clustered = []
    for d in sorted(dims, key=lambda x: float(x['value'])):
        if not clustered:
            clustered.append(d)
            continue
        prev = clustered[-1]
        if abs(float(d['value']) - float(prev['value'])) < tol:
            if d['confidence'] > prev['confidence']:
                clustered[-1] = d
        else:
            clustered.append(d)
    return clustered

# =========================
# Dimension Inference
# =========================
def infer_dimensions(words, circles, dim_lines, page_no):
    dims = []
    used = set()
    idx = 1

    for w in words:
        if not any(ch.isdigit() for ch in w['text']):
            continue
        if not is_valid_dimension(w['text']):
            continue

        value = w['text']
        dtype = 'linear'

        # Diameter: circle + dimension line + radius consistency
        for c in circles:
            dx = float(w['cx']) - float(c['x'])
            dy = float(w['cy']) - float(c['y'])
            dist_center = np.hypot(dx, dy)

            if abs(dist_center - c['r']) < c['r'] * 0.6:
                for l in dim_lines:
                    lx1, ly1, lx2, ly2 = l
                    d_line = min(
                        np.hypot(w['cx'] - lx1, w['cy'] - ly1),
                        np.hypot(w['cx'] - lx2, w['cy'] - ly2)
                    )
                    if d_line < 50:
                        dtype = 'diameter'
                        break

        key = (page_no, dtype, value)
        if key in used:
            continue
        used.add(key)

        dims.append({
            'id': idx,
            'page': page_no,
            'dim_type': dtype,
            'value': value,
            'unit': 'mm',
            'confidence': round(w['confidence'], 3)
        })
        idx += 1

    return cluster_dimensions(dims)

# =========================
# Flask Routes
# =========================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    temp_dir = tempfile.gettempdir()
    path = os.path.join(temp_dir, secure_filename(file.filename))
    file.save(path)

    images = convert_pdf_to_images(path) if path.lower().endswith('.pdf') else [cv2.imread(path)]
    all_dims = []

    for i, img in enumerate(images, start=1):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        circles = detect_circles(gray)
        dim_lines = detect_dimension_lines(gray)
        words = run_paddle_ocr(img)

        dims = infer_dimensions(words, circles, dim_lines, i)
        all_dims.extend(dims)

    df = pd.DataFrame(all_dims)
    df.sort_values(['page', 'dim_type'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['id'] = df.index + 1

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    excel = f'dimensions_{ts}.xlsx'
    csv = f'dimensions_{ts}.csv'

    df.to_excel(os.path.join(temp_dir, excel), index=False)
    df.to_csv(os.path.join(temp_dir, csv), index=False)

    return jsonify({
        'preview': df.head(30).to_dict(orient='records'),
        'excel_file': excel,
        'csv_file': csv
    })

@app.route('/download/<filename>')
def download(filename):
    path = os.path.join(tempfile.gettempdir(), secure_filename(filename))
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
