from flask import Flask, request, jsonify
from loan_predictor_ml.loan_prediction import predict_loan_status


from flask_cors import CORS  # Enable CORS for React frontend

app = Flask(__name__)
CORS(app)  # Allow React frontend to access API

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        user_input = request.json  # Receive JSON from React
        result = predict_loan_status(user_input)
        
        # Ensure the returned result is a JSON-compatible response
        return jsonify({"status": "Success", "prediction": result})
    except Exception as e:
        return jsonify({'status': 'Error', 'reason': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)  # Set port to 3000


