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

# ==== New User Data (replace with actual input) ====
new_data = pd.DataFrame({
    'person_age': [35],
    'person_gender': ['male'],
    'person_education': ['high_school'],
    'person_income': [50000],
    'person_emp_exp': [5],
    'person_home_ownership': ['rent'],
    'loan_amnt': [10000],
    'loan_intent': ['education'],
    'loan_int_rate': [12.5],
    'loan_percent_income': [0.2],
    'cb_person_cred_hist_length': [10],
    'credit_score': [700],
    'previous_loan_defaults_on_file': ['no']
})

# ==== Encode Categorical Features (Handle Unseen Labels) ====
for col in label_encoders:
    if col in new_data.columns:
        le = label_encoders[col]
        new_data[col] = new_data[col].apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )
print("✅ Categorical features encoded.")

# ==== Predict Loan Eligibility ====
prediction = rf_model.predict(new_data)
print("\n🔍 Loan Prediction Result:")
if prediction[0] == 1:
    print("✅ Loan Approved!")
else:
    print("❌ Loan Rejected!")
    # Create SHAP Explainer to Explain the Decision
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(new_data)
    rej={}
    # Get SHAP values for rejected prediction correctly
    for i in range(len(shap_values[0])):
        if(shap_values[0][i][0]>shap_values[0][i][1]):
          rej[new_data.columns[i]]=shap_values[0][i][0]-shap_values[0][i][1]
    sorted_dict = dict(sorted(rej.items(), key=lambda item: item[1], reverse=True))
    print("Reasons for Rejection:")
    rejection=[]
    for feature, value in list(sorted_dict.items())[:3]:
      rejection.append(feature)
    print(rejection)