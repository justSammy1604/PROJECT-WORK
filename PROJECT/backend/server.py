#To add flask code here. 
# Maybe add all the code in one file only 

from flask import Flask, request, jsonify
from app import rag_pipeline, response

app = Flask(__name__)

data = 1 # Terrence, you place your Crawler and the text it extracts here in this var.
rag_chain = rag_pipeline(data)

@app.route('/query', methods=['POST'])
def query():
  try:
    data = request.json
    query = data.get('query')
    if not query:
      return jsonify({'error':'Query Not Provided'}),400
    answer = response(query,rag_chain)
    return jsonify({'answer':answer})
  except Exception as e:
     return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4200, debug=True)
