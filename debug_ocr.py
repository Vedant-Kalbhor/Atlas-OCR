"""Dump raw text for page 6 of DRAWINGS PDF using high-res and low thresholds."""
import os
os.environ['USE_TORCH'] = '1'
import cv2
import numpy as np
import fitz # PyMuPDF
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

def dump_page(pdf_path, pno):
    doc_pdf = fitz.open(pdf_path)
    page = doc_pdf[pno-1]
    pix = page.get_pixmap(dpi=600)
    img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR if pix.n==4 else cv2.COLOR_RGB2BGR)
    
    model = ocr_predictor(det_arch='db_resnet50', reco_arch='parseq', pretrained=True,
                          assume_straight_pages=False, straighten_pages=True, 
                          export_as_straight_boxes=True, detect_orientation=True)
    model.det_predictor.model.postprocessor.bin_thresh = 0.05
    model.det_predictor.model.postprocessor.box_thresh = 0.05
    if os.environ.get("USE_TORCH"): import torch; model = model.cuda()

    doc = DocumentFile.from_images(cv2.imencode('.png', img_bgr)[1].tobytes())
    res = model(doc)
    json_out = res.export()
    
    print(f"--- PAGE {pno} RAW TEXT ---")
    for block in json_out['pages'][0]['blocks']:
        for line in block['lines']:
            line_txt = ' '.join([w['value'] for w in line['words']])
            line_conf = np.mean([w['confidence'] for w in line['words']])
            print(f"[{line_conf:.3f}] \"{line_txt}\"")

if __name__ == "__main__":
    dump_page('DRAWINGS ON BASIC FEATURES.pdf', 6)
    dump_page('DRAWINGS ON BASIC FEATURES.pdf', 4)
