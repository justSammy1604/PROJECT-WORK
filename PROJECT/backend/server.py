from flask import Flask, request, jsonify
from flask_cors import CORS 
from app import rag_pipeline, query_response 

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

data = 'data'  # We can also add crawled_data file as input here. 
rag_chain = rag_pipeline(data)

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.json
        query = data.get('query')
        if not query:
            return jsonify({'error': 'Query Not Provided'}), 400
        answer = query_response(query, rag_chain)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4200, debug=True)
