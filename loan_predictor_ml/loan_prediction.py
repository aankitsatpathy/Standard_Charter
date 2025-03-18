import pandas as pd
import joblib
import shap

# ==== File Paths ====
MODEL_PATH = 'best_random_forest_model.pkl'
ENCODERS_PATH = 'label_encoders.pkl'

# ==== Load Model & Label Encoders ====
try:
    rf_model = joblib.load(MODEL_PATH)
    label_encoders = joblib.load(ENCODERS_PATH)
    print("✅ Model and encoders loaded successfully.")
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    exit(1)

# ==== Prediction Function ====
def predict_loan_status(input_data_dict):
    """
    Predict loan status and provide reasons for rejection using SHAP.
    Args:
        input_data_dict (dict): Dictionary of user inputs.
    Returns:
        dict: {
            'status': 'Approved' / 'Rejected',
            'reason': 'Eligible' / List of top rejection reasons
        }
    """
    try:
        # Convert dict to DataFrame (single-row)
        user_df = pd.DataFrame([input_data_dict])

        # Encode categorical features
        for col in label_encoders:
            if col in user_df.columns:
                le = label_encoders[col]
                user_df[col] = user_df[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

        # Predict
        prediction = rf_model.predict(user_df)[0]

        if prediction == 1:
            return {'status': 'Approved', 'reason': 'Meets eligibility criteria.'}
        else:
            # SHAP Explanation for Rejection
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(user_df)

            # Compute SHAP contribution differences
            rej = {}
            for i, col in enumerate(user_df.columns):
                shap_diff = shap_values[0][0][i] - shap_values[1][0][i]
                if shap_diff > 0:
                    rej[col] = shap_diff

            # Sort top rejection reasons
            sorted_rej = dict(sorted(rej.items(), key=lambda item: item[1], reverse=True))
            top_reasons = list(sorted_rej.keys())[:3]  # Top 3 reasons

            return {'status': 'Rejected', 'reason': top_reasons}

    except Exception as e:
        return {'status': 'Error', 'reason': str(e)}
