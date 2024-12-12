from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import time
from app import rag_pipeline, query_response

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# data and RAG pipeline setup here
data = 1  # Terrence, you place your Crawler and the text it extracts here in this var.
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

@socketio.on('message')
def handle_message(message):
    # This is where you would integrate with your AI model
    # we'll just echo the message with a delay
    words = message.split()
    for word in words:
        time.sleep(0.5)  # Simulate processing time
        emit('message', word)

    # Send a final response
    time.sleep(1)
    emit('message', "This is the AI's response to: " + message)

if __name__ == 'main':
    # Running both Flask and SocketIO
    socketio.run(app, host='0.0.0.0', port=4200, debug=True)
