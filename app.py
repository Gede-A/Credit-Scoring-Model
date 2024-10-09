from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Step 1: Initialize Flask App
app = Flask(__name__)

# Step 2: Load the trained model (from Task 4)
model = joblib.load('trained_model.pkl')  # Make sure you have the trained model saved in this path

# Step 3: Preprocessing - Define the Scaler (Assuming you used scaling in training)
scaler = StandardScaler()

# Step 4: Define API endpoints

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input data from the API request
        input_data = request.get_json()

        # Convert input data to DataFrame
        df = pd.DataFrame(input_data)

        # Apply any necessary preprocessing (assuming input needs scaling)
        scaled_data = scaler.transform(df)

        # Step 5: Make predictions
        predictions = model.predict(scaled_data)

        # Return predictions as JSON
        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})

# Step 6: Running the Flask app
if __name__ == '__main__':
    app.run(debug=True)
