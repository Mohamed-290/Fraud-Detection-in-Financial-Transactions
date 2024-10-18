from flask import Flask, render_template, request
import mlflow.pyfunc
import logging
import numpy as np
import re
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Set logging level to DEBUG
logging.basicConfig(level=logging.DEBUG)

mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Load models
dt_model_name = "FraudDetectionModel"
lstm_model_name = "FraudDetectionModel_LSTM"

try:
    dt_model = mlflow.pyfunc.load_model(f"models:/{dt_model_name}/1")
    print("Decision Tree model loaded successfully.")
except Exception as e:
    print(f"Error loading Decision Tree model: {e}")

try:
    lstm_model = mlflow.pyfunc.load_model(f"models:/{lstm_model_name}/1")
    print("LSTM model loaded successfully.")
except Exception as e:
    print(f"Error loading LSTM model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit_data():
    # Get form data
    step = request.form.get('step', type=int)
    amount = request.form.get('amount', type=float)
    payment_type = request.form.get('type', type=int)
    action_sequence = request.form.get('action_sequence', type=str)
    model_choice = request.form.get('model', type=str)

    # Prepare the input for prediction based on the model choice
    if model_choice == "ml":  # Adjusted to match the HTML form
        # Prepare input for Decision Tree model
        data = np.array([[step, amount, payment_type]])
        prediction = dt_model.predict(data)
    else:  # For LSTM model
        processed_action_sequence = preprocess_action_sequence(action_sequence)

        # Tokenize the action sequence (You need to implement the word to index mapping)
        action_sequence_array = tokenize_sequence(processed_action_sequence)

        max_len = 10  # Adjust according to your model's input shape
        action_sequence_padded = pad_sequences([action_sequence_array], maxlen=max_len, padding='post')

        # Combine padded sequence with other features
        features = np.array([[step, amount, payment_type]])
        data = np.concatenate((action_sequence_padded, features), axis=1)

        # Make prediction using the LSTM model
        prediction = lstm_model.predict(data)

    # Debugging output to check prediction shape
    print(f"Prediction: {prediction}, Shape: {prediction.shape}")

    # Convert prediction to binary based on its structure
    if isinstance(prediction, np.ndarray) and prediction.ndim == 2:
        result_text = "Fraud" if prediction[0][0] >= 0.5 else "Not Fraud"
    else:
        result_text = "Fraud" if prediction >= 0.5 else "Not Fraud"  # Assuming scalar output

    return render_template('index.html', result=result_text)

def preprocess_action_sequence(sequence):
    segments = sequence.split('/')
    processed_segments = []
    for segment in segments:
        segment = segment.lower()  # Lowercase the text
        segment = re.sub(r'[^a-z\s]', '', segment)  # Remove non-alphabetical characters
        segment = re.sub(r'\s+', ' ', segment).strip()  # Remove extra spaces
        processed_segments.append(segment)
    return processed_segments  # Return the processed segments or further process them as needed

def tokenize_sequence(processed_sequence):
    # Implement a mapping of words to integers (this should ideally be your model's training vocabulary)
    word_to_index = {
        'payment': 1,
        'transfer': 2,
        'cash_in': 3,
        'cash_out': 4,
        'debit': 5,
        # Add more mappings based on your model's vocabulary
    }
    
    tokenized_sequence = []
    for segment in processed_sequence:
        if segment in word_to_index:
            tokenized_sequence.append(word_to_index[segment])
        else:
            tokenized_sequence.append(0)  # Handle unknown words with a default value

    return tokenized_sequence

if __name__ == '__main__':
    app.run(debug=True, port=8080)
