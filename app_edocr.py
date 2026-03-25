"""
Atlas-OCR (eDOCr Engine)
========================
Alternative OCR engine using eDOCr — a specialised OCR system for
mechanical engineering drawings built on keras-ocr.

eDOCr advantages over generic OCR:
  • Purpose-trained alphabet that includes Ø, ±, °, ⌀ and GD&T symbols
  • Built-in drawing segmentation (info-block / GD&T / dimensions / frame)
  • Automatic rotation handling for angled dimension text
  • Tolerance parsing (H7, h6, ±0.05, upper/lower bounds)
  • Patch-based detection with agglomerative clustering

Pipeline:
  1. PDF → image via pdf2image (poppler) or PyMuPDF
  2. eDOCr box_tree  → find rectangles (info-block, frame, GD&T)
  3. eDOCr img_process → isolate dimension region
  4. eDOCr pipeline_dimensions → detect + recognise dimension text
  5. Post-process → classify & export to XLSX / CSV

Requirements (install in a conda env):
  pip install eDOCr opencv-python flask pandas openpyxl pdf2image
  (on Windows you also need poppler: choco install poppler  OR  conda install -c conda-forge poppler)

Usage:
  python app_edocr.py                     # starts Flask on http://localhost:5001
  python app_edocr.py --file Sample.pdf   # CLI mode, prints results to stdout
"""

import os
import re
import sys
import cv2
import string
import logging
import tempfile
import argparse
import numpy as np

# ──── NumPy 2.0 compatibility shim for imgaug (eDOCr dependency) ────
# imgaug uses np.sctypes which was removed in NumPy 2.0.
# We monkey-patch it back so eDOCr can import cleanly.
if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'int':     [np.int8, np.int16, np.int32, np.int64],
        'uint':    [np.uint8, np.uint16, np.uint32, np.uint64],
        'float':   [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128],
        'others':  [bool, object, bytes, str, np.void],
    }
if not hasattr(np, 'bool'):
    np.bool = np.bool_
if not hasattr(np, 'int'):
    np.int = np.int_
if not hasattr(np, 'float'):
    np.float = np.float64
if not hasattr(np, 'complex'):
    np.complex = np.complex128
if not hasattr(np, 'object'):
    np.object = np.object_
if not hasattr(np, 'str'):
    np.str = np.str_

import pandas as pd
from datetime import datetime

# ──────────────────── CONFIG ────────────────────
CLUSTER_THRESHOLD = 20       # px distance for grouping nearby detections
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("atlas-edocr")

# ──────────────────── eDOCr ALPHABET & MODELS ────────────────────
# eDOCr ships with pre-trained models for:
#   - dimensions:  digits + engineering chars (M, R, Ø, ±, °, etc.)
#   - info-block:  alphanumeric + punctuation
#   - GD&T:        geometric tolerance symbols
GDT_SYMBOLS  = '⏤⏥○⌭⌒⌓⏊∠⫽⌯⌖◎↗⌰'
FCF_SYMBOLS  = 'ⒺⒻⓁⓂⓅⓈⓉⓊ'
EXTRA_CHARS  = '(),.+-±:/°"⌀'

ALPHABET_DIM       = string.digits + 'AaBCDRGHhMmnx' + EXTRA_CHARS
ALPHABET_INFOBLOCK = string.digits + string.ascii_letters + ',.:-/'
ALPHABET_GDT       = string.digits + ',.⌀ABCD' + GDT_SYMBOLS

# Color palette for the annotated mask image
COLOR_PALETTE = {
    'infoblock':  (180, 220, 250),
    'gdts':       (94,  204, 243),
    'dimensions': (93,  206, 175),
    'frame':      (167, 234, 82),
    'flag':       (241, 65,  36),
}

# ──────────────────── LAZY GLOBALS ────────────────────
_models_loaded = False
_model_dim = None
_model_info = None
_model_gdt = None


def _ensure_edocr():
    """Import eDOCr and download models on first call."""
    global _models_loaded, _model_dim, _model_info, _model_gdt
    if _models_loaded:
        return

    try:
        from eDOCr import keras_ocr
        log.info("Downloading / verifying eDOCr models …")
        _model_info = keras_ocr.tools.download_and_verify(
            url="https://github.com/javvi51/eDOCr/releases/download/v1.0.0/recognizer_infoblock.h5",
            filename="recognizer_infoblock.h5",
            sha256="e0a317e07ce75235f67460713cf1b559e02ae2282303eec4a1f76ef211fcb8e8",
        )
        _model_dim = keras_ocr.tools.download_and_verify(
            url="https://github.com/javvi51/eDOCr/releases/download/v1.0.0/recognizer_dimensions.h5",
            filename="recognizer_dimensions.h5",
            sha256="a1c27296b1757234a90780ccc831762638b9e66faf69171f5520817130e05b8f",
        )
        _model_gdt = keras_ocr.tools.download_and_verify(
            url="https://github.com/javvi51/eDOCr/releases/download/v1.0.0/recognizer_gdts.h5",
            filename="recognizer_gdts.h5",
            sha256="58acf6292a43ff90a344111729fc70cf35f0c3ca4dfd622016456c0b29ef2a46",
        )
        _models_loaded = True
        log.info("eDOCr models ready ✓")
    except ImportError:
        raise ImportError(
            "eDOCr is not installed.  Install it with:\n"
            "  pip install eDOCr\n"
            "See: https://github.com/javvi51/eDOCr"
        )


# ──────────────────── IMAGE LOADING ────────────────────

def load_image(file_path):
    """
    Load a PDF or image file into a list of BGR numpy arrays (one per page).
    Uses PyMuPDF (fitz) for PDF rendering at 300 DPI.
    """
    images = []
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pdf':
        try:
            # Try PyMuPDF first (already installed in the project)
            import fitz
            doc = fitz.open(file_path)
            for page in doc:
                pix = page.get_pixmap(dpi=300)
                img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                if pix.n == 4:
                    bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
                else:
                    bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                images.append(bgr)
            doc.close()
        except ImportError:
            # Fallback: pdf2image (requires poppler)
            from pdf2image import convert_from_path
            pil_imgs = convert_from_path(file_path, dpi=300)
            for pil_img in pil_imgs:
                arr = np.array(pil_img)
                images.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    else:
        img = cv2.imread(file_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {file_path}")
        images.append(img)

    return images


# ──────────────────── DIMENSION CLASSIFICATION ────────────────────

def classify_edocr_pred(pred_dict):
    """
    Take the raw eDOCr prediction dict and produce a standardised row.

    eDOCr pred_dict keys:
        ID, type, flag, nominal, value, tolerance,
        upper_bound, lower_bound
    """
    nominal = str(pred_dict.get('nominal', ''))
    value   = str(pred_dict.get('value', ''))
    dtype   = str(pred_dict.get('type', 'Length'))
    tol     = pred_dict.get('tolerance', '')
    upper   = pred_dict.get('upper_bound', '')
    lower   = pred_dict.get('lower_bound', '')
    flag    = pred_dict.get('flag', False)

    # Map eDOCr type → our type labels
    type_map = {
        'Length':    'linear',
        'Thread':   'thread',
        'Angle':    'angle',
        'Roughness':'roughness',
    }
    dim_type = type_map.get(dtype, 'linear')

    # Detect additional types from nominal text
    if '⌀' in nominal or 'Ø' in nominal:
        dim_type = 'diameter'
    elif nominal.upper().startswith('R') and re.match(r'R\s*\d', nominal, re.IGNORECASE):
        dim_type = 'radius'
    elif 'CB' in nominal.upper():
        dim_type = 'counterbore'
    elif re.search(r'\d+\s*HOLES?', nominal, re.IGNORECASE):
        dim_type = 'hole_pattern'

    # Build feature name
    clean_val = re.sub(r'[^0-9.xX×±]', '', value) if value else nominal
    feature = f"{dim_type}_{clean_val}" if clean_val else f"{dim_type}_{nominal}"

    # Build tolerance string
    tol_str = ''
    if upper and lower:
        tol_str = f"+{upper}/{lower}"
    elif tol:
        tol_str = str(tol)

    # Unit
    unit = 'deg' if dim_type == 'angle' else 'mm'

    return {
        'feature':    feature,
        'value':      value if value else nominal,
        'unit':       unit,
        'type':       dim_type,
        'nominal':    nominal,
        'tolerance':  tol_str,
        'flagged':    '⚠' if flag else '',
    }


# ──────────────────── MAIN PROCESSING ────────────────────

def process_drawing_edocr(filepath, filename, cluster_t=CLUSTER_THRESHOLD):
    """
    Full eDOCr pipeline: segmentation → OCR → classification → export.

    Returns: (result_dict, error_msg, excel_filename, csv_filename)
    """
    _ensure_edocr()
    from eDOCr import tools
    from skimage import io as skio

    cleanup_paths = []
    try:
        images = load_image(filepath)
        all_data = []
        seen = set()
        global_id = 1

        for page_idx, img in enumerate(images):
            page_no = page_idx + 1
            log.info(f"  eDOCr: Processing page {page_no} …")

            # ── 1. Segmentation: find rectangles ──
            try:
                class_list, img_boxes = tools.box_tree.findrect(img)
            except Exception as e:
                log.warning(f"  Page {page_no}: box_tree failed ({e}), using full image")
                class_list = []
                img_boxes = img.copy()

            # ── 2. Separate info-blocks, GD&T, frame ──
            try:
                boxes_info, gdt_boxes, cl_frame, process_img = tools.img_process.process_rect(class_list, img)
            except Exception as e:
                log.warning(f"  Page {page_no}: process_rect failed ({e}), using full image")
                boxes_info, gdt_boxes, cl_frame = [], [], None
                process_img = img.copy()

            # Save processed image temporarily (eDOCr pipeline_dimensions needs a file path)
            proc_path = os.path.join(tempfile.gettempdir(), f"edocr_proc_p{page_no}.jpg")
            skio.imsave(proc_path, process_img)
            cleanup_paths.append(proc_path)

            # ── 3. OCR info-blocks ──
            try:
                infoblock_dict = tools.pipeline_infoblock.read_infoblocks(
                    boxes_info, img, ALPHABET_INFOBLOCK, _model_info
                )
                log.info(f"    Info-blocks: {len(infoblock_dict)} items")
            except Exception as e:
                log.warning(f"    Info-block OCR failed: {e}")
                infoblock_dict = []

            # ── 4. OCR GD&T ──
            try:
                gdt_dict = tools.pipeline_gdts.read_gdtbox1(
                    gdt_boxes, ALPHABET_GDT, _model_gdt, ALPHABET_DIM, _model_dim
                )
                log.info(f"    GD&T: {len(gdt_dict)} items")
            except Exception as e:
                log.warning(f"    GD&T OCR failed: {e}")
                gdt_dict = []

            # ── 5. OCR Dimensions (the main prize) ──
            try:
                dimension_dict = tools.pipeline_dimensions.read_dimensions(
                    proc_path, ALPHABET_DIM, _model_dim, cluster_t
                )
                log.info(f"    Dimensions (segmented): {len(dimension_dict)} items")
            except Exception as e:
                log.warning(f"    Dimension OCR failed: {e}")
                dimension_dict = []

            # ── 5b. Fallback: if segmented image yields 0 dims, try full image ──
            if not dimension_dict:
                log.info(f"    Falling back to full-page OCR...")
                full_path = os.path.join(tempfile.gettempdir(), f"edocr_full_p{page_no}.jpg")
                skio.imsave(full_path, img)
                cleanup_paths.append(full_path)
                try:
                    dimension_dict = tools.pipeline_dimensions.read_dimensions(
                        full_path, ALPHABET_DIM, _model_dim, cluster_t
                    )
                    log.info(f"    Dimensions (fallback full-page): {len(dimension_dict)} items")
                except Exception as e:
                    log.warning(f"    Full-page dimension OCR also failed: {e}")
                    dimension_dict = []

            # ── 6. Collect results ──

            # Dimensions
            for dim in dimension_dict:
                pred = dim.get('pred', {})
                row = classify_edocr_pred(pred)
                key = (page_no, row['value'], row['type'])
                if key not in seen and row['value']:
                    seen.add(key)
                    row['id']    = str(global_id)
                    row['notes'] = f"page {page_no}"
                    all_data.append(row)
                    global_id += 1

            # GD&T entries
            for gdt in gdt_dict:
                pred = gdt.get('text', {})
                nominal = str(pred.get('nominal', ''))
                if nominal:
                    key = (page_no, nominal, 'gdt')
                    if key not in seen:
                        seen.add(key)
                        all_data.append({
                            'id':        str(global_id),
                            'feature':   f"gdt_{nominal}",
                            'value':     nominal,
                            'unit':      'mm',
                            'type':      'gdt',
                            'nominal':   nominal,
                            'tolerance': str(pred.get('tolerance', '')),
                            'flagged':   '⚠' if pred.get('flag') else '',
                            'notes':     f"page {page_no}",
                        })
                        global_id += 1

            # Info-block entries (usually title / material / revision)
            for ib in infoblock_dict:
                pred = ib.get('text', {})
                nominal = str(pred.get('nominal', ''))
                if nominal:
                    key = (page_no, nominal, 'info')
                    if key not in seen:
                        seen.add(key)
                        all_data.append({
                            'id':        str(global_id),
                            'feature':   f"info_{nominal}",
                            'value':     nominal,
                            'unit':      '-',
                            'type':      'info_block',
                            'nominal':   nominal,
                            'tolerance': '',
                            'flagged':   '',
                            'notes':     f"page {page_no}",
                        })
                        global_id += 1

        if not all_data:
            return None, "No dimensions found by eDOCr.", None, None

        # ─── Export ───
        temp_dir = tempfile.gettempdir()
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        excel_name = f'edocr_dims_{ts}.xlsx'
        csv_name   = f'edocr_dims_{ts}.csv'
        excel_path = os.path.join(temp_dir, excel_name)
        csv_path   = os.path.join(temp_dir, csv_name)

        df = pd.DataFrame(all_data)
        cols = ['id', 'feature', 'value', 'unit', 'type', 'nominal', 'tolerance', 'flagged', 'notes']
        existing = [c for c in cols if c in df.columns]
        df[existing].to_excel(excel_path, index=False)
        df[existing].to_csv(csv_path, index=False)

        log.info(f"  Exported {len(all_data)} rows → {excel_name}")
        return {"columns": existing, "data": all_data}, None, excel_name, csv_name

    except Exception as e:
        log.exception("eDOCr pipeline error")
        return None, str(e), None, None

    finally:
        for p in cleanup_paths:
            try:
                os.remove(p)
            except:
                pass


# ──────────────────── FLASK APP ────────────────────

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    temp_dir = tempfile.gettempdir()
    filename = secure_filename(file.filename)
    filepath = os.path.join(temp_dir, filename)
    file.save(filepath)

    try:
        result, error, excel_name, csv_name = process_drawing_edocr(filepath, filename)
        if error:
            return jsonify({'error': error}), 500
        return jsonify({
            'preview': result,
            'excel_file': excel_name,
            'csv_file': csv_name,
        })
    except Exception as e:
        log.exception("Error processing file")
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            os.remove(filepath)
        except:
            pass


@app.route('/download/<filename>')
def download(filename):
    safe = secure_filename(filename)
    path = os.path.join(tempfile.gettempdir(), safe)
    if not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404
    return send_file(path, as_attachment=True, download_name=safe)


# ──────────────────── CLI MODE ────────────────────

def cli_main():
    """Run from command line: python app_edocr.py --file drawing.pdf"""
    parser = argparse.ArgumentParser(description="Atlas-OCR (eDOCr engine)")
    parser.add_argument('--file', type=str, help='Path to the drawing (PDF/image)')
    parser.add_argument('--cluster', type=int, default=CLUSTER_THRESHOLD,
                        help=f'Cluster threshold in px (default: {CLUSTER_THRESHOLD})')
    parser.add_argument('--serve', action='store_true', help='Start Flask server')
    args = parser.parse_args()

    if args.file:
        log.info(f"Processing: {args.file}")
        result, error, excel, csv_f = process_drawing_edocr(args.file, os.path.basename(args.file), args.cluster)
        if error:
            print(f"ERROR: {error}")
            sys.exit(1)

        print(f"\n{'='*70}")
        print(f" eDOCr Extraction Results — {len(result['data'])} dimensions")
        print(f"{'='*70}")

        # Group by page
        by_page = {}
        for row in result['data']:
            p = row.get('notes', 'unknown')
            by_page.setdefault(p, []).append(row)

        for page in sorted(by_page.keys()):
            print(f"\n--- {page} ---")
            for row in by_page[page]:
                flag = row.get('flagged', '')
                tol  = row.get('tolerance', '')
                print(f"  {flag:2s} {row['type']:18s} {row['value']:12s} "
                      f"{'('+tol+')' if tol else '':15s} [{row['nominal']}]")

        if excel:
            print(f"\n  Excel: {os.path.join(tempfile.gettempdir(), excel)}")
        if csv_f:
            print(f"  CSV:   {os.path.join(tempfile.gettempdir(), csv_f)}")

    elif args.serve or not args.file:
        log.info("="*60)
        log.info("  Atlas-OCR (eDOCr Engine)")
        log.info("  Powered by eDOCr — keras-ocr for engineering drawings")
        log.info("="*60)
        app.run(debug=True, port=5001, use_reloader=False)


if __name__ == '__main__':
    cli_main()
