import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ======= PATHS ========
input_csv = "..."  # replace with your_input_articles
output_csv = "..." # replace with articles_with_vad_emotions

# ====== Load pretrained VAD model =======
MODEL_NAME = "bhadresh-savani/bert-base-uncased-emotion-vader"  # Replace if needed

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def predict_vad(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    vad = outputs.logits.squeeze().cpu().numpy()
    vad = np.clip(vad, 0, 1)
    return vad  # [Valence, Arousal, Dominance]

def vad_to_emotion_category(vad):
    valence, arousal, dominance = vad
    # hand-crafted from NRC VAD Lexicon + Mehrabian’s PAD emotion vectors
    if 0.4 < valence < 0.6 and 0.4 < arousal < 0.6:
        return 0  # Neutral
    if valence >= 0.6 and arousal >= 0.6:
        return 1  # Happy
    if valence < 0.4 and arousal < 0.4:
        return 2  # Sad
    if valence >= 0.5 and arousal >= 0.8:
        return 3  # Surprise
    if valence < 0.4 and arousal >= 0.7:
        return 4  # Fear
    if valence < 0.3 and dominance < 0.3:
        return 5  # Disgust
    if valence < 0.3 and arousal >= 0.5:
        return 6  # Anger
    if dominance < 0.4 and valence < 0.5:
        return 7  # Contempt
    return 0

EMOTION_LABELS = {
    0: "Neutral",
    1: "Happy",
    2: "Sad",
    3: "Surprise",
    4: "Fear",
    5: "Disgust",
    6: "Anger",
    7: "Contempt"
}

def analyze_text(text):
    vad = predict_vad(text)
    category = vad_to_emotion_category(vad)
    return vad, category

def process_csv(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)

    valences = []
    arousals = []
    dominances = []
    categories = []
    category_names = []

    for idx, row in df.iterrows():
        text = row['article_text']
        vad, cat = analyze_text(text)
        valences.append(vad[0])
        arousals.append(vad[1])
        dominances.append(vad[2])
        categories.append(cat)
        category_names.append(EMOTION_LABELS[cat])
        print(f"Processed article {idx+1}/{len(df)}: {row['article_name']} -> {EMOTION_LABELS[cat]}")

    df['valence'] = valences
    df['arousal'] = arousals
    df['dominance'] = dominances
    df['emotion_category_id'] = categories
    df['emotion_category_name'] = category_names

    df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")

if __name__ == "__main__":
    process_csv(input_csv, output_csv)
