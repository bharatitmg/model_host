from flask import Flask, request, jsonify
import os
import keras
from huggingface_hub import login
import numpy as np

app = Flask(__name__)

# Set Hugging Face API token and log in
HUGGINGFACE_TOKEN = "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"  # Replace with your actual Hugging Face API token
login(token=HUGGINGFACE_TOKEN)  # Authenticate with Hugging Face

os.environ["KERAS_BACKEND"] = "tensorflow"

# Load the model directly using the hf:// protocol
model = keras.saving.load_model("hf://beejaytmg/ai_tic_tac_toe")

@app.route('/')
def home():
    return "Hello, World!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['board']
        if len(data) != 9:
            return jsonify({'error': 'Invalid board state. Must have 9 elements.'}), 400
        prediction = model.predict(np.array([data]))
        move = int(np.argmax(prediction))
        return jsonify({'move': move})
    except KeyError:
        return jsonify({'error': 'Missing "board" key in JSON.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
