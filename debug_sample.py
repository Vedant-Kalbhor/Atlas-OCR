"""Debug what docTR is seeing for Sample.pdf"""
import os
os.environ['USE_TORCH'] = '1'
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import fitz
import numpy as np
import cv2

# Manual render to 600 DPI to match app.py
doc = fitz.open("Sample.pdf")
page = doc[0]
pix = page.get_pixmap(dpi=600)
img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR if pix.n==4 else cv2.COLOR_RGB2BGR)

model = ocr_predictor(det_arch='db_resnet50', reco_arch='parseq', pretrained=True)
model.det_predictor.model.postprocessor.bin_thresh = 0.05
model.det_predictor.model.postprocessor.box_thresh = 0.05
import torch; model = model.cuda()

res = model(DocumentFile.from_images(cv2.imencode('.png', img_bgr)[1].tobytes()))
json_out = res.export()

for block in json_out['pages'][0]['blocks']:
    for line in block['lines']:
        print(f"[{np.mean([w['confidence'] for w in line['words']]):.2f}] {' '.join([w['value'] for w in line['words']])}")
