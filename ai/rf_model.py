import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def train_rf_model(feature_dict, errors):
    X = []
    y = []

    for elem_id, features in feature_dict.items():
        X.append(features)
        y.append(1 if elem_id in errors else 0)

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("ML MODEL PERFORMANCE")
    print(classification_report(y_test, y_pred))

    return model


def predict_failure_probability(model, feature_dict):
    probabilities = {}

    for elem_id, features in feature_dict.items():
        prob = model.predict_proba([features])[0][1]
        probabilities[elem_id] = prob

    return probabilities

