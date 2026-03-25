"""Detailed debug for page 4."""
import os
os.environ['USE_TORCH'] = '1'
import sys
from app import process_drawing

result, error, excel, csv_f = process_drawing('DRAWINGS ON BASIC FEATURES.pdf', 'DRAWINGS_ON_BASIC_FEATURES.pdf')
if error:
    print(f'ERROR: {error}')
else:
    print(f"{'Page':10} | {'Type':15} | {'Value':10} | {'Feature':20}")
    print("-" * 60)
    for row in result['data']:
        if row['notes'] == 'page 4':
            print(f"{row['notes']:10} | {row['type']:15} | {row['value']:10} | {row['feature']:20}")
