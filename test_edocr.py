"""
Diagnostic test script for eDOCr pipeline on Sample.pdf with better logging.
"""
import os, sys, cv2
import numpy as np
import logging

# Set up logging to stdout
logging.basicConfig(level=logging.INFO, format='%(message)s')

# NumPy 2.0 compat shim
if not hasattr(np, 'sctypes'):
    np.sctypes = {'int': [np.int8, np.int16, np.int32, np.int64], 'uint': [np.uint8, np.uint16, np.uint32, np.uint64], 'float': [np.float16, np.float32, np.float64], 'complex': [np.complex64, np.complex128], 'others': [bool, object, bytes, str, np.void]}
if not hasattr(np, 'bool'): np.bool = np.bool_
if not hasattr(np, 'int'): np.int = np.int_
if not hasattr(np, 'float'): np.float = np.float64
if not hasattr(np, 'complex'): np.complex = np.complex128
if not hasattr(np, 'object'): np.object = np.object_
if not hasattr(np, 'str'): np.str = np.str_

try:
    import string
    import fitz
    from skimage import io as skio
    from eDOCr import tools, keras_ocr
except Exception as e:
    print(f"IMPORT ERROR: {e}")
    sys.exit(1)

def run_test():
    # Load image from PDF
    print("Loading Sample.pdf at 600 DPI...")
    try:
        doc = fitz.open("Sample.pdf")
        pix = doc[0].get_pixmap(dpi=600)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
        doc.close()
        print(f"Image shape: {img.shape}")
    except Exception as e:
        print(f"FAILED TO LOAD PDF: {e}")
        return

    # Alphabet and models
    EXTRA = '(),.+-±:/°"⌀'
    alphabet_dim = string.digits + 'AaBCDRGHhMmnx' + EXTRA
    try:
        model_dim = keras_ocr.tools.download_and_verify(
            url="https://github.com/javvi51/eDOCr/releases/download/v1.0.0/recognizer_dimensions.h5",
            filename="recognizer_dimensions.h5",
            sha256="a1c27296b1757234a90780ccc831762638b9e66faf69171f5520817130e05b8f",
        )
        print(f"Model path: {model_dim}")
    except Exception as e:
        print(f"FAILED TO DOWNLOAD MODEL: {e}")
        return

    # Step 1: findrect (Segmentation)
    print("\n--- Step 1: Segmentation (box_tree) ---")
    try:
        class_list, img_boxes = tools.box_tree.findrect(img)
        print(f"  Found {len(class_list)} rectangle groups")
    except Exception as e:
        print(f"  Segmentation error: {e}")
        class_list = []

    # Step 2: process_rect
    print("\n--- Step 2: Processing regions ---")
    try:
        boxes_info, gdt_boxes, cl_frame, process_img = tools.img_process.process_rect(class_list, img)
        print(f"  Info blocks: {len(boxes_info)}, GDT: {len(gdt_boxes)}")
        print(f"  Frame rect: {cl_frame}")
    except Exception as e:
        print(f"  Region processing error: {e}")
        process_img = img.copy()

    # Step 3: OCR on segmented content
    print("\n--- Step 3: Dimension OCR (segmented) ---")
    proc_path = "edocr_test_proc.jpg"
    skio.imsave(proc_path, process_img)
    try:
        dims = tools.pipeline_dimensions.read_dimensions(proc_path, alphabet_dim, model_dim, 20)
        print(f"  Found {len(dims)} dimensions in segmented image")
        for d in dims:
            p = d.get('pred', {})
            print(f"    - [{p.get('type')}] '{p.get('nominal')}' (val: {p.get('value')})")
    except Exception as e:
        print(f"  Segmented OCR error: {e}")
        dims = []

    # Step 4: Fallback to full page OCR
    if not dims:
        print("\n--- Step 4: Dimension OCR (full page fallback) ---")
        full_path = "edocr_test_full.jpg"
        skio.imsave(full_path, img)
        try:
            dims = tools.pipeline_dimensions.read_dimensions(full_path, alphabet_dim, model_dim, 20)
            print(f"  Found {len(dims)} dimensions in full page")
            for d in dims:
                p = d.get('pred', {})
                print(f"    - [{p.get('type')}] '{p.get('nominal')}' (val: {p.get('value')})")
        except Exception as e:
            print(f"  Full page OCR error: {e}")

    print("\nDONE")

if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        print(f"CRITICAL SCRIPT ERROR: {e}")
        import traceback
        traceback.print_exc()
