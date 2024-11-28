from ast import Import
from flask import Flask, request, jsonify
from flask_cors import CORS
import json 
import os

app = Flask(__name__)
CORS(app) 

#function to write the logs in a JSON file
def write_logs(data,file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    # Append the new log to the list
    existing_data.append({
        'textContent': data.get('textContent'),
        'timestamp': data.get('timestamp'),
        'title': data.get('title'),
        'element': data.get('element')
    })

    # Save updated data back to the file
    with open(file_path, 'w') as file:
        json.dump(existing_data, file, indent=4)
            
            
@app.route('/log_click', methods=['POST'])
#function to record the clicks
def log_click():
    data = request.json  # Get the JSON data from the request
    if not data or 'timestamp' not in data or 'element' not in data:
        return jsonify({'status': 'error', 'message': 'Invalid data'}), 400

    file_path = "C:/Users/TERREL BRAGANCA/chatbot/intern-project-stuff/demo_website/smartschoolmis/Session1.JSON" #Enter the path of the JSON file where the logs need to be stored
    write_logs(data,file_path)
    
    processed_data = {
        'status': 'success',
        'message': f"Received click on {data.get('element', 'unknown element')}",
        'received_at': data['timestamp']
    }
    

    return jsonify(processed_data)

if __name__ == '__main__':
    app.run(debug=True)
