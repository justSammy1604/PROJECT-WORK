from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import time

# Placeholder for your RAG pipeline and query response (you'll need to import or define these)
def rag_pipeline(data):
    # Implement your RAG pipeline initialization
    return None

def query_response(query, rag_chain):
    # Implement your query response logic
    return f"Response to: {query}"

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Terrence, you place your Crawler and the text it extracts here in this var.
data = 1 

# Initialize RAG chain
rag_chain = rag_pipeline(data)

# REST API Endpoint for Queries
@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.json
        query = data.get('query')
        if not query:
            return jsonify({'error':'Query Not Provided'}), 400
        
        answer = query_response(query, rag_chain)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# WebSocket Message Handler
@socketio.on('message')
def handle_message(message):
    try:
        # Process message through RAG pipeline
        words = message.split()
        
        # Simulate streaming response
        for word in words:
            time.sleep(0.5)  # Simulate processing time
            emit('message', word)
        
        # Generate full response using RAG
        full_response = query_response(message, rag_chain)
        
        # Send final response
        time.sleep(1)
        emit('message', full_response)
    
    except Exception as e:
        emit('error', {'error': str(e)})

# Main entry point
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=4200, debug=True)
