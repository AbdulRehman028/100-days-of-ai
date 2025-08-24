import os
import pandas as pd
import json  # For JSON handling
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['JSON_FOLDER'] = 'json_files'  # New folder for saving JSON files
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['JSON_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith('.csv'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Handle NaN values by replacing them with `None`
            df = df.where(pd.notnull(df), None)  # Replace NaN/None with None
            
            # Convert to JSON
            json_data = df.to_dict(orient='records')
            
            # Save JSON data to a file
            json_filename = file.filename.replace('.csv', '.json')
            json_file_path = os.path.join(app.config['JSON_FOLDER'], json_filename)
            with open(json_file_path, 'w') as json_file:
                json.dump(json_data, json_file)

            # Preview the first 5 records of JSON data
            preview_data = json_data[:5]  # Show only the first 5 records for preview

            # Provide the download URL
            download_url = f'/download/{json_filename}'
            
            return jsonify({
                'message': 'File converted successfully!',
                'download_url': download_url,
                'json_preview': preview_data  # Send JSON preview data to the frontend
            })

        except Exception as e:
            return jsonify({'error': f"An error occurred: {str(e)}"}), 500

    return jsonify({'error': 'Invalid file format. Please upload a CSV.'}), 400

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['JSON_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
