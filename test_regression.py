"""Verify regession on Sample.pdf (Target: 10, 410, 133, 169, 122)"""
import os
os.environ['USE_TORCH'] = '1'
import sys
from app import process_drawing

result, error, excel, csv_f = process_drawing('Sample.pdf', 'Sample.pdf')
if error:
    print(f'ERROR: {error}')
else:
    print("\nEXTRACTED DIMENSIONS:")
    found = {row['value']: row['type'] for row in result['data']}
    for k, v in found.items():
        print(f"  {k:10} | {v}")
    
    expected = {'10', '410', '133', '169', '122'}
    found_vals = set(found.keys())
    print(f"\nMissing:  {expected - found_vals}")
    print(f"Extra:    {found_vals - expected}")
