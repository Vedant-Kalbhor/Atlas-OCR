"""Test refined app.py on DRAWINGS ON BASIC FEATURES.pdf"""
import os
os.environ['USE_TORCH'] = '1'
import sys
sys.path.insert(0, '.')
from app import process_drawing
import json

result, error, excel, csv_f = process_drawing('DRAWINGS ON BASIC FEATURES.pdf', 'DRAWINGS_ON_BASIC_FEATURES.pdf')
if error:
    print(f'ERROR: {error}')
else:
    print(f"\n{'='*60}")
    print(f"Total: {len(result['data'])} dimensions")
    print(f"{'='*60}")
    
    by_page = {}
    for row in result['data']:
        p = row['notes']
        by_page.setdefault(p, []).append(row)
    
    for page in sorted(by_page.keys()):
        print(f"\n--- {page} ---")
        for row in by_page[page]:
            print(f"  [{row['confidence']}] {row['type']:18s} {row['value']:10s} ({row['feature']})")
