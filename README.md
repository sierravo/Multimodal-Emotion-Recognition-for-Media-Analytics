# Facial Emotion Recognition using BReG-NeXt and Facial Landmarks

This project builds a multi-modal facial emotion recognition system combining:
- CNN-based image classification (BReG-NeXt)
- Landmark-based feature modeling (XGBoost)

The goal is to compare and combine structured and unstructured approaches to improve emotion classification performance on AffectNet.

This repository reimplements a prior collaborative project from scratch, with additional improvements including:
- modular pipeline design
- multi-modal aggregation
- reproducible training scripts

## Acknowledgement

This project builds on prior collaborative work. I acknowledge guidance and contributions from:
- Dr. Siddhartha Dalal
- Dr. Michael Lesk
- Wesley Yuan
- Vikki Sui

## Key Features
- Multi-modal emotion classification (image + landmarks + text)
- Deep learning (BReG-NeXt) + classical ML (XGBoost)
- End-to-end pipeline from data ingestion → prediction → aggregation
- Modular architecture for experimentation

## Results

| Model                 | Accuracy |
|-----------------------|----------|
| BReG-NeXt (CNN)       | 59%      |
| Landmark + XGBoost    | 57.9%    |
| Combined (Softmax)    | 77%      |


## Key Learnings
- Deep learning vs. engineered features:
  CNN models learn robust visual features directly from images, while landmark-based models depend on handcrafted geometry and are more sensitive to preprocessing.
- Label noise impacts performance:
  AffectNet contains ambiguous labels, which limits achievable accuracy regardless of model choice.
- Feature representation differences:
  Landmark features capture structure only, while pixel-based models capture both structure and texture, leading to stronger performance.
- Multi-modal aggregation:
  Combining outputs from different models improves prediction stability by leveraging complementary information.
- Modular pipeline design:
  Separating pipelines improves maintainability and enables independent experimentation.
- Reproducibility constraints:
  Large datasets and legacy dependencies (e.g., TensorFlow 1.x) complicate environment setup and replication.

## Limitations
- Dataset limitations:
  AffectNet is imbalanced and contains subjective labels.
- HTML parsing is brittle if the NY Times page structure changes.
- Landmark model constraints:
  Geometric features lack texture information, limiting performance.
- Preprocessing dependency:
  Errors in face detection or landmark extraction degrade results.
- Legacy dependencies:
  TensorFlow 1.x introduces compatibility issues.
- Limited evaluation:
  Evaluation is restricted to AffectNet with no cross-dataset validation.
- No deployment:
  The system is designed for offline use only.
- Manual setup required:
  Paths and data must be configured manually.


---

## Project Structure

- scripts/
  - pixel_based/ → CNN models
  - landmark_preprocessing/ → feature engineering + XGBoost
  - nlp/ → text classification
  - scraping/ → data collection
- manuscript/ → research reference

```
image-emotion-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                             # Place dataset or processed dataset
feature files here (not included)
├── outputs/                          # Model predictions (generated)
├── models/                           # Saved trained models (generated)
├── scripts/
│   ├── nypost/
│   │   ├── scraper_utils.py          # Utilities for scraping
│   │   ├── scraper_main.py           # Main scraper logic
│   │   └── scraper_links.py          # Link extraction for scraper
│   ├── nytimes/
│   │   ├── article_fetcher.py        # Fetch articles from NYTimes
│   │   ├── config.py                 # API keys and settings
│   │   ├── main.py                   # Entry point for NYTimes scraper
│   │   ├── nyt_api.py                # API wrapper for NYTimes
│   │   └── rate_limiter.py           # Handles API rate limiting
│   ├── landmark_preprocessing/
│   │   ├── run_xgb_landmark.py       # XGBoost landmark classifier runner
│   │   ├── data_loader.py            # Load and preprocess landmark data
│   │   ├── feature_engineering.py    # Extract features from landmarks
│   │   ├── image_processing.py       # Image pre-processing utilities
│   │   ├── landmarks_utils.py        # Landmark-specific helper functions
│   │   ├── main.py                   # Landmark preprocessing pipeline entry
│   │   ├── models.py                 # Landmark-based model definitions
│   │   ├── train.py                  # Training scripts for models
│   │   └── utils.py                  # General utilities
│   ├── pixel_based/
│   │   ├── bregnext_loader.py        # Loader for BRegNext model
│   │   ├── data_loader.py            # Load pixel-based image data
│   │   ├── emotion_model_loader.py   # Load emotion classification model
│   │   ├── main_pipeline.py          # Main pixel-based processing pipeline
│   │   ├── model_runner.py           # Run pixel-based models
│   │   ├── pretrained_bregnext_loader.py  # Load pretrained BRegNext weights
│   │   └── transforms.py             # Image transforms and augmentations
│   ├── text_emotions_classifier.py   # Text-based emotion classification script
│   └── softmax_agg_predictions.py    # Softmax aggregation of predictions
└── manuscript/
    └── Finding_Emotions_in_Faces_A_Meta_Classifier.pdf

```

---

## Requirements

You can install dependencies using:

```bash
pip install -r requirements.txt
```
- TensorFlow 1.x compatibility is required
- Checkpoint files are external
- Operation names are model-specific

---

## Data Availability

The original dataset (e.g., AffectNet) is not included in this repository due to size and licensing restrictions. Please request access from the official source:

👉 [http://mohammadmahoor.com/affectnet/](http://mohammadmahoor.com/affectnet/)

Output results were not stored due to storage constraints. Once the dataset is downloaded and placed into the appropriate folders, you can re-run the pipeline to reproduce the results.

## Outputs

- `outputs/`: CSV files with model predictions and probabilities
- `models/`: Serialized trained models (.joblib)

---

## How to Use

Run training with default settings (expects data in `data/` folder):

```bash
python scripts/landmark_preprocessing/main.py
```
To specify a custom data directory:
```bash
python scripts/landmark_preprocessing/main.py --data_dir /path/to/data

```
To run specific feature files:
```bash
python scripts/landmark_preprocessing/main.py --feature_files new_features.csv
```

1. Facial Landmark Emotion Classification
Use handcrafted features (e.g., inter-point distances, center distances) to classify facial emotions:

```bash
python scripts/landmark_preprocessing/main.py \
  --data_dir data \
  --feature_files new_features.csv \
  --output_dir outputs
```

To run only the XGBoost classifier:

```bash
python run_xgb_landmark.py
Input: Precomputed .csv files of features
```

Output: CSV of predictions with class probabilities

2. Pixel-Based Emotion Classification
Use BRegNext or other CNN models to classify emotions directly from facial images:

Pretrained models are not included.

Provide paths via CLI:

python main_pipeline.py \
  --affectnet_ckpt path/to/affectnet.pth \
  --songfan_ckpt path/to/songfan.pth \
  --image_dir path/to/images

```bash
python scripts/pixel_based/main.py \
  --image_dir data/images/ \
  --bregnext_ckpt checkpoints/categorical_attempt_3/ \
  --affectnet_ckpt models/affectnet.pth \
  --songfan_ckpt models/songfan.pth \
  --example_dir examplespth \
  --songfan_ckpt models/songfan.pth
```
Input: Folder of cropped face images
Output: CSV with emotion predictions and softmax probabilities

3. Text-Based Emotion Classification
Classify emotions from text content (e.g., from news or dialogue):

```bash
python text_emotions_classifier.py \
  --input_csv data/articles.csv \
  --output_csv outputs/articles_with_emotions.csv \
  --text_column article_text \
  --name_column article_name \
  --batch_size 16
```

4. Web Article Scraping (Optional)
Scrape from New York Post:

```bash
python scripts/nypost/main.py \
  --year 2023 \
  --month 5 \
  --output_dir data/nypost
```

Scrape from New York Times:

```bash
python scripts/nytimes/main.py \
  --start_date 2022-01-01 \
  --end_date 2022-03-01 \
  --output_dir outputs/nyt
```
Note: Add your NYTimes API key to config.py

5. Aggregating Predictions
Merge outputs from text, landmark, and image pipelines using a softmax-weighted strategy:

```bash
python scripts/python softmax_agg_predictions.py \
  --xgb_input outputs/xgb_predictions.csv \
  --dl_input outputs/bregnext_predictions.csv \
  --output_csv outputs/combined_predictions.csv
```

Input: Softmax probability CSVs from each model
Output: Final aggregated emotion prediction


---

## Notes

Make sure to update path variables in the scripts to match your directory structure.

---

## License

This project is for research and educational purposes.

---

