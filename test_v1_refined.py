"""
Unified Test Runner for app-v1_refined.py
Benchmarks Sample.pdf and DRAWINGS ON BASIC FEATURES.pdf
"""
import os
import sys

# Ensure current dir is in path
sys.path.append('.')

# Import process_drawing from the updated app-v1.py
# (Since it has a hyphen in the filename, we use importlib)
import importlib
try:
    module = importlib.import_module('app-v1')
    process_drawing = module.process_drawing
except Exception as e:
    print(f"FAILED TO IMPORT app-v1.py: {e}")
    sys.exit(1)

def run_bench(pdf_name, expected_dims=None):
    print(f"\n{'='*50}")
    print(f" TESTING: {pdf_name}")
    print(f"{'='*50}")
    
    if not os.path.exists(pdf_name):
        print(f"  FILE NOT FOUND: {pdf_name}")
        return

    res, err, ex, cs = process_drawing(pdf_name, os.path.basename(pdf_name))
    if err:
        print(f"  ERROR: {err}")
        return

    data = res['data']
    print(f"  TOTAL DIMENSIONS EXTRACTED: {len(data)}")
    
    # Sort and group by page
    by_page = {}
    for row in data:
        p = row['page']
        by_page.setdefault(p, []).append(row['value'])
    
    for p in sorted(by_page.keys()):
        print(f"    Page {p}: {', '.join(by_page[p])}")

    if expected_dims:
        found = set(map(str, [r['value'] for r in data]))
        missing = [str(e) for e in expected_dims if str(e) not in found]
        extra = [str(f) for f in found if str(f) not in list(map(str, expected_dims))]
        
        print(f"\n  COMPLIANCE:")
        print(f"    Extra:   {extra if extra else 'None'}")
        print(f"    Missing: {missing if missing else 'None'}")
        if not missing:
            print(f"    SUCCESS: All expected dimensions captured!")

if __name__ == "__main__":
    # Test 1: Sample.pdf
    run_bench("Sample.pdf", expected_dims=["10", "410", "133", "169", "122"])
    
    # Test 2: DRAWINGS ON BASIC FEATURES.pdf
    # (Just page 1 for now to verify logic)
    run_bench("DRAWINGS ON BASIC FEATURES.pdf")
