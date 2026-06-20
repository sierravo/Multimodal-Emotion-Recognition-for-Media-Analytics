import pandas as pd
import argparse 
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ====== Load pretrained VAD model =======
MODEL_NAME = "bhadresh-savani/bert-base-uncased-emotion-vader"  # Replace if needed

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

def load_model(model_name=MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

def predict_vad_batch(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}    

    with torch.no_grad():
        outputs = model(**inputs)

    vad = outputs.logits.detach().cpu().numpy()
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

def process_csv(input_csv_path, output_csv_path, text_column="article_text", name_column = "article_name", batch_size=16):
    tokenizer, model, device = load_model()

    df = pd.read_csv(input_csv_path)

    if text_column not in df.columns:
        raise ValueError(
            f"Missing required text column: '{text_column}'. "
            f"Available columns: {list(df.columns)}"
        )

    texts = df[text_column].fillna("").astype(str).tolist()
    if name_column is not None and name_column not in df.columns:
        print(f"Optional name column '{name_column}' not found; continuing without it.")

    all_vad = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        vad_batch = predict_vad_batch(batch_texts, tokenizer, model, device)
        all_vad.extend(vad_batch)

        print(f"Processed batch {i // batch_size + 1}")

    all_vad = np.array(all_vad)

    valence = all_vad[:, 0]
    arousal = all_vad[:, 1]
    dominance = all_vad[:, 2]

    categories = [vad_to_emotion_category(v) for v in all_vad]
    category_names = [EMOTION_LABELS[c] for c in categories]

    result_df = df.copy()

    result_df["valence"] = valence
    result_df["arousal"] = arousal
    result_df["dominance"] = dominance
    result_df["emotion_category_id"] = categories
    result_df["emotion_category_name"] = category_names

    import os
    out_dir = os.path.dirname(output_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    result_df.to_csv(output_csv_path, index=False)
    print(f"Saved to {output_csv_path}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify article text into emotion categories using VAD scores."
    )
    parser.add_argument("--input_csv", required=True, help="Path to input CSV")
    parser.add_argument("--output_csv", required=True, help="Path to output CSV")
    parser.add_argument("--text_column", default="article_text", help="Name of text column")
    parser.add_argument("--name_column", default="article_name", help="Optional name/title column")
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()

def main():
    args = parse_args()
    process_csv(
        input_csv_path=args.input_csv,
        output_csv_path=args.output_csv,
        text_column=args.text_column,
        name_column=args.name_column,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
