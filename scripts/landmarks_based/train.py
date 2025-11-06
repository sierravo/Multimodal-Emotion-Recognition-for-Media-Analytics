from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import save_predictions

def run_classifiers(df, dataset_name, classifiers):
    """
    Train multiple classifiers on the dataset and save prediction outputs.
    Returns a list of tuples: (model_name, accuracy, dataset_name, trained_model)
    """
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    results = []

    for model_name, clf in classifiers.items():
        print(f"\nTraining {model_name} on {dataset_name} dataset...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        acc = accuracy_score(y_test, y_pred)
        save_predictions(y_test, y_pred, y_proba, dataset_name, model_name)
        results.append((model_name, acc, dataset_name, clf))

    return results
