import pandas as pd
import joblib

affectnet_labels = {
    0: 'Neutral',
    1: 'Happy',
    2: 'Sad',
    3: 'Surprise',
    4: 'Fear',
    5: 'Disgust',
    6: 'Anger',
    7: 'Contempt'
}

def save_predictions(y_test, y_pred, y_proba, dataset_name, model_name, output_dir="/content"):
    """
    Save prediction probabilities along with true and predicted labels as CSV.
    """
    df_preds = pd.DataFrame(y_proba, columns=[f"prob_{affectnet_labels[i]}" for i in range(len(affectnet_labels))])
    df_preds.insert(0, "true_label", y_test.reset_index(drop=True))
    df_preds.insert(1, "predicted_label", y_pred)
    filename = f"{output_dir}/{dataset_name.replace(' ', '_')}_{model_name.replace(' ', '_')}_affectnet_predictions.csv"
    df_preds.to_csv(filename, index=False)
    print(f"Saved predictions to {filename}")

def save_model(model, model_name, dataset_name, output_dir="/content"):
    """
    Save the trained model to disk.
    """
    filename = f"{output_dir}/best_model_{model_name.replace(' ', '_')}_{dataset_name.replace(' ', '_')}.joblib"
    joblib.dump(model, filename)
    print(f"Best model saved to: {filename}")
