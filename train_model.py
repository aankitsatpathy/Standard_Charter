import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==== File Paths ====
DATA_PATH = 'loan_data.csv'  # Change this if needed
MODEL_PATH = 'best_random_forest_model.pkl'
ENCODERS_PATH = 'label_encoders.pkl'

# ==== Load Data ====
try:
    data = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded dataset from '{DATA_PATH}'")
except FileNotFoundError:
    print(f"❌ Error: File '{DATA_PATH}' not found.")
    exit(1)

# ==== Encode Categorical Features ====
categorical_cols = [
    'person_gender',
    'person_education',
    'person_home_ownership',
    'loan_intent',
    'previous_loan_defaults_on_file'
]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le
print("✅ Encoded categorical features")

# Save encoders for future use
joblib.dump(label_encoders, ENCODERS_PATH)
print(f"💾 Label encoders saved to '{ENCODERS_PATH}'")

# ==== Prepare Features and Target ====
X = data.drop('loan_status', axis=1)
y = data['loan_status']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✅ Split data into train ({len(X_train)}) and test ({len(X_test)}) sets")

# ==== Model Training with Grid Search ====
rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print("🔍 Starting Grid Search for hyperparameter tuning...")
grid_search = GridSearchCV(
    rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2
)
grid_search.fit(X_train, y_train)

# Get the best model
best_rf = grid_search.best_estimator_
print(f"✅ Best Hyperparameters: {grid_search.best_params_}")

# Save the trained model
joblib.dump(best_rf, MODEL_PATH)
print(f"💾 Best Random Forest model saved to '{MODEL_PATH}'")

# ==== Model Evaluation ====
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\n📊 Model Evaluation Results:")
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report)

print("\n✅ Training complete. Model and encoders are ready for deployment.")
