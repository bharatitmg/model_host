from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np

app = Flask(__name__)
model = load_model('model_v4_1.keras')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['board']
        if len(data) != 9:
            return jsonify({'error': 'Invalid board state. Must have 9 elements.'}), 400
        prediction = model.predict(np.array([data]))
        move = int(np.argmax(prediction))  # Convert to int
        return jsonify({'move': move})
    except KeyError:
        return jsonify({'error': 'Missing "board" key in JSON.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
