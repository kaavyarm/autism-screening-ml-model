import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_split_data():
    # load cleaned dataset
    df = pd.read_csv("ml/autism_cleaned.csv")

    # separate target
    y = df["Class/ASD"]
    X = df.drop(columns=["Class/ASD"])

    # one-hot encode categorical columns
    X = pd.get_dummies(X, drop_first=True)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler