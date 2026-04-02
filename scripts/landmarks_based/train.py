from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from utils import save_predictions


def run_classifiers(df, dataset_name, classifiers):
    X = df.drop(columns=["target", "img_path"])
    y = df["target"]

    # Stratified split (important for imbalanced data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=0,
        stratify=y
    )

    results = []

    for name, model in classifiers.items():
        print(f"\nTraining {name} on {dataset_name}...")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        print(f"{name} Accuracy: {acc:.4f}")
        print(f"{name} Macro F1: {f1:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Probabilities (if available)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
        else:
            y_proba = None

        save_predictions(y_test, y_pred, y_proba, dataset_name, name)

        results.append((name, f1, dataset_name, model))  # use F1 for comparison

    return results