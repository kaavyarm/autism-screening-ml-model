from ml.preprocess import load_and_split_data, scale_data
from ml.evaluate import evaluate_model

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import joblib


# Step 1: load data
X_train, X_test, y_train, y_test = load_and_split_data()

# Step 2: scale features
X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

# Step 3: train logistic regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)

print("Logistic Regression trained")

# Step 4: train random forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train_scaled, y_train)

print("Random Forest trained")

# Step 5: evaluate models
log_acc, log_roc, _ = evaluate_model(log_model, X_test_scaled, y_test)
rf_acc, rf_roc, rf_cm = evaluate_model(rf_model, X_test_scaled, y_test)

print("\n=== MODEL PERFORMANCE ===")
print(f"Logistic Accuracy: {log_acc:.3f}")
print(f"Random Forest Accuracy: {rf_acc:.3f}")

print(f"\nLogistic ROC AUC: {log_roc:.3f}")
print(f"Random Forest ROC AUC: {rf_roc:.3f}")

print("\nConfusion Matrix (Random Forest):")
print(rf_cm)


# Step 6: choose best model (we’ll use RF)
final_model = rf_model


# Step 7: save artifacts
joblib.dump(final_model, "ml/model.pkl")
joblib.dump(scaler, "ml/scaler.pkl")

# VERY IMPORTANT: save feature columns for inference
joblib.dump(X_train.columns.tolist(), "ml/columns.pkl")

print("\nModel, scaler, and columns saved")