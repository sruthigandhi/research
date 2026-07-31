import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_dataset(path):
    df = pd.read_csv(path, index_col='Date', parse_dates=True)
    X = df.drop(columns=['target'])
    y = df['target']
    return X, y


def train_and_evaluate(name, X_train, X_test, y_train, y_test):
    print(f"\n{'=' * 60}")
    print(f"{name.upper()} DATASET")
    print(f"{'=' * 60}")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape:  {X_test.shape}")

    models = {
        "logistic_regression": LogisticRegression(max_iter=5000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        print(f"\n[{model_name}] Accuracy: {acc:.3f}")
        print(classification_report(y_test, preds, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, preds))


if __name__ == "__main__":
    datasets = [
        ("clean", "train_data_clean.csv", "test_data_clean.csv"),
        ("leaky", "train_data_leaky.csv", "test_data_leaky.csv"),
    ]

    for name, train_path, test_path in datasets:
        X_train, y_train = load_dataset(train_path)
        X_test, y_test = load_dataset(test_path)
        train_and_evaluate(name, X_train, X_test, y_train, y_test)
