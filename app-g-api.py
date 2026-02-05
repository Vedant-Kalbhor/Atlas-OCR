from flask import Flask, render_template, request, jsonify, send_file
import google.generativeai as genai
import os
import pandas as pd
import json
from werkzeug.utils import secure_filename
import tempfile
from datetime import datetime

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure Gemini API
GOOGLE_API_KEY = "AIzaSyDXEEgvA1Sva_ijO0P9cuvovh1TLxIzNkY"
genai.configure(api_key=GOOGLE_API_KEY)

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp'}

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
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload PDF or image files.'}), 400
    
    try:
        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)
        file.save(filepath)
        
        # Upload file to Gemini
        uploaded_file = genai.upload_file(filepath)
        
        # Create model and generate response
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = """Analyze this engineering drawing and extract ALL dimensions mentioned in the document.

You MUST respond with ONLY a valid JSON object, no other text before or after. The JSON should have this exact structure:

{
    "columns": ["id", "column1", "column2", ...],
    "data": [
        {"id": "1", "column1": "value1", "column2": "value2", ...},
        {"id": "2", "column1": "value1", "column2": "value2", ...}
    ]
}

Instructions:
1. Create appropriate column names based on the dimensions found (e.g., "inner_diameter_mm", "outer_diameter_mm", "length_mm", "width_mm", "height_mm", "thickness_mm", etc.)
2. Always include "id" as the first column
3. Extract all dimensions and organize them as rows of data
4. Include units in the column names (e.g., _mm, _inches, _cm)
5. If multiple parts/components are shown, create separate rows for each
6. Use "N/A" for missing values
7. Ensure all data rows have values for all columns

Example response:
{
    "columns": ["id", "inner_diameter_mm", "outer_diameter_mm", "length_mm"],
    "data": [
        {"id": "1", "inner_diameter_mm": "25.4", "outer_diameter_mm": "50.8", "length_mm": "100"},
        {"id": "2", "inner_diameter_mm": "30.0", "outer_diameter_mm": "60.0", "length_mm": "150"}
    ]
}

Remember: Respond ONLY with the JSON object, nothing else."""
        
        response = model.generate_content([uploaded_file, prompt])
        
        # Clean up uploaded file
        os.remove(filepath)
        
        # Parse the JSON response
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith('```'):
            response_text = response_text.split('```')[1]
            if response_text.startswith('json'):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        dimensions_data = json.loads(response_text)
        
        # Create Excel file
        excel_filename = f'dimensions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        excel_path = os.path.join(temp_dir, excel_filename)
        
        df = pd.DataFrame(dimensions_data['data'])
        df.to_excel(excel_path, index=False, engine='openpyxl')
        
        return jsonify({
            'dimensions': response.text,
            'excel_file': excel_filename,
            'preview': dimensions_data
        })
        
    except json.JSONDecodeError as e:
        return jsonify({'error': f'Error parsing AI response as JSON: {str(e)}\n\nRaw response: {response.text}'}), 500
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/download/<filename>')
def download(filename):
    try:
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, secure_filename(filename))
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(
            filepath,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        return jsonify({'error': f'Error downloading file: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)