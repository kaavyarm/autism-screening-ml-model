# ml/evaluate.py

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    roc = roc_auc_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    return accuracy, roc, cm