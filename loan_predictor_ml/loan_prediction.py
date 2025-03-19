import pandas as pd
import joblib
import shap

# Load model and encoders
MODEL_PATH = 'loan_predictor_ml/best_random_forest_model.pkl'
ENCODERS_PATH = 'loan_predictor_ml/label_encoders.pkl'

rf_model = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENCODERS_PATH)

def predict_loan_status(user_data_dict):
    new_data = pd.DataFrame([user_data_dict])

    # Encode categorical features
    for col in label_encoders:
        if col in new_data.columns:
            le = label_encoders[col]
            new_data[col] = new_data[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    prediction = rf_model.predict(new_data)

    if prediction[0] == 1:
        return {'status': 'Approved'}
    else:
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(new_data)
        rej = {}
        for i in range(len(shap_values[0][0])):
            diff = shap_values[0][0][i] - shap_values[1][0][i]
            if diff > 0:
                rej[new_data.columns[i]] = diff
        sorted_rej = sorted(rej.items(), key=lambda item: item[1], reverse=True)
        top_reasons = [feature for feature, _ in sorted_rej[:3]]
        return {'status': 'Rejected', 'reason': top_reasons}
