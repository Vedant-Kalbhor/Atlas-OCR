"""Check Coords of 6910 in Sample.pdf."""
import os, sys, fitz, cv2, numpy as np, importlib
sys.path.append('.')
module = importlib.import_module('app-v1')
from doctr.io import DocumentFile

model = module.get_ocr_model()
doc_doctr = DocumentFile.from_images("sample_600.png")
res = model(doc_doctr).export()

for pg in res['pages']:
    for block in pg['blocks']:
        for line in block['lines']:
            txt = " ".join([w['value'] for w in line['words']])
            if "69" in txt or "169" in txt or "10" in txt:
                print(f"RAW TEXT: '{txt}'")
                for w in line['words']:
                    geom = w['geometry']
                    print(f"  WORD: '{w['value']}' | Geom: {geom}")
