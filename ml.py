import pandas as pd
import numpy as np
import joblib  # To save & load model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Data (Change filename if needed)
data = pd.read_csv('/content/loan_data.csv')  

# Encode Categorical Features
categorical_cols = ['person_gender', 'person_education', 'person_home_ownership', 
                    'loan_intent', 'previous_loan_defaults_on_file']

label_encoders = {}  # Store encoders for later decoding if needed
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Save encoder

# Save Label Encoders for Future Use
joblib.dump(label_encoders, "label_encoders.pkl")

# Define Features & Target Variable
X = data.drop('loan_status', axis=1)  
y = data['loan_status']  

# Split Data into Train & Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest Classifier
rf = RandomForestClassifier(random_state=42)

# Define Hyperparameter Grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform Grid Search for Hyperparameter Tuning
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get Best Model from Grid Search
best_rf = grid_search.best_estimator_

# Save the Best Model
joblib.dump(best_rf, "best_random_forest_model.pkl")

# Make Predictions
y_pred = best_rf.predict(X_test)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print Results
print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report)

print("\n✅ Model saved as 'best_random_forest_model.pkl'")
print("✅ Label encoders saved as 'label_encoders.pkl'")