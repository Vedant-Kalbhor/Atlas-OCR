from flask import Flask, render_template, request, jsonify, send_file
import os
import cv2
import numpy as np
import pandas as pd
import tempfile
from datetime import datetime
from pdf2image import convert_from_path
from werkzeug.utils import secure_filename
import easyocr

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}

# =========================
# EasyOCR Initialization
# =========================
reader = easyocr.Reader(['en'])

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
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,
        param1=120,
        param2=30,
        minRadius=10,
        maxRadius=400
    )
    results = []
    if circles is not None:
        for c in np.uint16(np.around(circles[0])):
            results.append({'x': int(c[0]), 'y': int(c[1]), 'r': int(c[2])})
    return results


def detect_dimension_lines(gray):
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,
                            minLineLength=60, maxLineGap=5)
    dim_lines = []

    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            length = np.hypot(x2 - x1, y2 - y1)

            if length > 60:
                dim_lines.append((x1, y1, x2, y2))

    return dim_lines


# =========================
# EasyOCR Based OCR
# =========================
def run_easyocr(image):
    results = reader.readtext(image)

    words = []

    for (bbox, text, conf) in results:

        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]

        cx = int(sum(x_coords) / 4)
        cy = int(sum(y_coords) / 4)

        words.append({
            'text': text.replace('Ø', '').replace('⌀', ''),
            'confidence': float(conf),
            'cx': cx,
            'cy': cy,
            'bbox': bbox
        })

    return words


# =========================
# Utility Filters
# =========================
def is_valid_dimension(text):
    try:
        val = float(text)
        return 0.1 < val < 10000
    except:
        return False


def remove_duplicates(dims, tol=2):
    unique = []

    for d in sorted(dims, key=lambda x: float(x['value'])):
        if not unique:
            unique.append(d)
            continue

        prev = unique[-1]
        if abs(float(prev['value']) - float(d['value'])) < tol:
            if d['confidence'] > prev['confidence']:
                unique[-1] = d
        else:
            unique.append(d)

    return unique


# =========================
# Dimension Inference using Bounding Boxes
# =========================
def infer_dimensions(words, circles, dim_lines, page_no):
    dims = []
    used = set()
    idx = 1

    for w in words:

        text = w['text']

        if not any(ch.isdigit() for ch in text):
            continue

        if not is_valid_dimension(text):
            continue

        dtype = "linear"

        # Check for diameter based on circle proximity
        for c in circles:
            dx = w['cx'] - c['x']
            dy = w['cy'] - c['y']

            if np.hypot(dx, dy) < c['r'] * 1.3:
                dtype = "diameter"
                break

        # If near dimension line -> likely linear
        for line in dim_lines:
            x1, y1, x2, y2 = line

            dist = min(
                np.hypot(w['cx'] - x1, w['cy'] - y1),
                np.hypot(w['cx'] - x2, w['cy'] - y2)
            )

            if dist < 50:
                dtype = "linear"
                break

        key = (page_no, dtype, text)

        if key in used:
            continue

        used.add(key)

        dims.append({
            'id': idx,
            'page': page_no,
            'dim_type': dtype,
            'value': text,
            'unit': 'mm',
            'confidence': round(w['confidence'], 3)
        })

        idx += 1

    return remove_duplicates(dims)


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
        words = run_easyocr(img)

        dims = infer_dimensions(words, circles, dim_lines, i)
        all_dims.extend(dims)

    df = pd.DataFrame(all_dims)

    df.sort_values(['page', 'dim_type'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['id'] = df.index + 1

    df_export = df.drop(columns=['confidence', 'unit', 'dim_type'], errors='ignore')

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    excel = f'dimensions_{ts}.xlsx'
    csv = f'dimensions_{ts}.csv'

    df_export.to_excel(os.path.join(temp_dir, excel), index=False)
    df_export.to_csv(os.path.join(temp_dir, csv), index=False)

    return jsonify({
        'preview': df_export.head(30).to_dict(orient='records'),
        'excel_file': excel,
        'csv_file': csv
    })


@app.route('/download/<filename>')
def download(filename):
    path = os.path.join(tempfile.gettempdir(), secure_filename(filename))
    return send_file(path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
