from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from chatbot_logic import ChatbotLogic
import os

app = Flask(__name__, static_folder='static')
CORS(app)

# Initialize chatbot logic
chatbot = ChatbotLogic()

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory(app.static_folder, path)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_question = data.get('question', '')
    
    if not user_question:
        return jsonify({'answer': "Please ask a question!"}), 400
    
    response = chatbot.get_response(user_question)
    return jsonify({'answer': response})

if __name__ == '__main__':
    # Ensure static directory exists
    if not os.path.exists('static'):
        os.makedirs('static')
    
    app.run(debug=True, port=5000)
