
from scripts.eda import load_data

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import joblib
import os
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load Dataset
df = load_data()

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

# Features and Target
X = df.drop("churn", axis=1)
y = df["churn"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# SVM Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        random_state=42
    ))
])

# Hyperparameter Grid
param_grid = {
    "svm__C": [0.1, 1, 10],
    "svm__kernel": ["linear", "rbf"],
    "svm__gamma": ["scale", "auto"]
}

# GridSearchCV Object
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

print("\nGridSearchCV Object Created!")

# Start Hyperparameter Tuning

print("\nStarting GridSearchCV Training...")

grid_search.fit(X_train, y_train)

print("\nGridSearchCV Training Completed!")

# Best Parameters
print("\nBest Parameters:")
print(grid_search.best_params_)

# Best Cross Validation Score
print("\nBest Cross Validation Accuracy:")
print(grid_search.best_score_)


# Best Model
best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)

# Test Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nTuned Model Test Accuracy:")
print(accuracy)

# Classification Report
print("\nTuned Model Classification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Retained", "Churned"],
    zero_division=0
))

# Confusion Matrix
print("\nTuned Model Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)


# Save Best Model

os.makedirs("models", exist_ok=True)

joblib.dump(
    best_model,
    "models/churn_model.joblib"
)

print("\nBest Model Saved Successfully!")