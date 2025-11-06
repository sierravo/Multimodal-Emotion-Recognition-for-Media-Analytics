import pandas as pd
import joblib
import os

# ==== CONFIG ====

MODEL_PATH = "/content/..."  # <-- Path to trained model
NEW_DATASET_PATH = "/content/..."  # <-- Path to new image feature data
OUTPUT_CSV_PATH = "/content/..." # <--- Path to prediction output file

# AffectNet label mapping (optional, for readability)
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

# ==== LOAD MODEL ====

print(f"Loading model from: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# ==== LOAD NEW DATA ====

print(f"Loading new dataset from: {NEW_DATASET_PATH}")
df_new = pd.read_csv(NEW_DATASET_PATH)

# If dataset has labels (e.g., first column), drop them for prediction
if df_new.columns[0].lower() in ['label', 'emotion', 'class']:
    X_new = df_new.iloc[:, 1:]
else:
    X_new = df_new

# ==== PREDICT ====

print("Predicting labels and probabilities...")
y_pred = model.predict(X_new)
y_proba = model.predict_proba(X_new)

# ==== OUTPUT ====

# Prepare DataFrame
df_output = pd.DataFrame(y_proba, columns=[f"prob_{affectnet_labels[i]}" for i in range(len(affectnet_labels))])
df_output.insert(0, "predicted_label", y_pred)
df_output.insert(1, "predicted_emotion", [affectnet_labels[label] for label in y_pred])

# Save to CSV
df_output.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"Predictions saved to: {OUTPUT_CSV_PATH}")
