# Multimodal Emotion Recognition for Media Analytics

This project is a business-facing data science reimplementation of a prior collaborative facial emotion recognition project. It explores how multiple data modalities — facial image pixels, facial landmarks, and article text — can be combined to estimate dominant emotional signals in media content.

The core machine learning task is 8-class emotion classification:

- Neutral
- Happy
- Sad
- Surprise
- Fear
- Disgust
- Anger
- Contempt

The project compares two facial emotion modeling approaches:

1. **Pixel-based deep learning** using BReG-NeXt / CNN-style image classification.
2. **Landmark-based machine learning** using engineered facial geometry features with XGBoost.

It also includes article scraping and text-emotion scoring components to support a broader media analytics workflow.

---

## Business Use Case

This project can support media, brand, marketing, and communications analytics workflows where teams want to understand emotional framing across article text and associated images.

Example business questions:

- Are articles about a topic framed more often with fear, anger, sadness, happiness, or neutrality?
- Do article images and article text communicate the same emotional signal?
- Can multimodal signals help flag emotionally charged content for editorial, brand-safety, or campaign-analysis review?
- Does combining image-based and landmark-based predictions improve classification reliability compared with either model alone?

This project is **not intended to infer a person’s true internal emotional state**. It estimates observable facial-expression signals and text-based emotional signals from available data.

---

## Project Objective

The goal is to compare and combine structured and unstructured approaches to emotion classification:

| Modality            | Input                          | Method                                     |
|---------------------|--------------------------------|--------------------------------------------|
| Facial landmarks    | 68 facial landmark coordinates | Engineered distances/angles + XGBoost      |
| Facial image pixels | Cropped facial images          | BReG-NeXt / CNN-based model                |
| Article text        | News/article text              | Transformer/VAD-based text emotion scoring |
| Ensemble            | Model probability outputs      | Softmax-style probability aggregation      |

The main data science question is:

> Can combining facial landmark features and pixel-based deep learning predictions improve emotion classification accuracy?

The accompanying manuscript found that the two facial models made different errors, which made model aggregation useful.

---

## Results Summary

The accompanying manuscript evaluated the facial emotion models on a balanced subset of manually annotated AffectNet images. After duplicate removal, the experiment used:

- **16,854 training images**
- **4,520 test images**
- **8 emotion categories**
- **12.5% random guessing baseline** for forced-choice classification

| Model                    | Accuracy | F1 Score | Description                                                              |
|--------------------------|---------:|---------:|--------------------------------------------------------------------------|
| Random Guessing Baseline | 12.5%    | —        | 8-class forced-choice baseline                                           |
| Landmark + XGBoost       | 58.1%    | 57.9%    | Uses 68 facial landmarks with engineered distance and angle features     |
| Pixel-Based BReG-NeXt    | ~58%     | —        | CNN-based model using full facial image pixels                           |
| Combined Meta-Classifier | 76.7%    | 76.8%    | Combines landmark and pixel-model probability outputs, then renormalizes |

The combined classifier improved performance because the landmark and pixel models made different errors. The manuscript reports that the two models had only fair agreement, with Cohen’s kappa of **30.14%**, supporting the use of an ensemble/meta-classifier approach.

**Important:** These metrics come from the accompanying manuscript. This repository reimplements and organizes the project pipelines. Full reproduction requires external datasets and model checkpoints that are not included in the repository.

---

## Key Findings

### 1. Landmark and pixel models were complementary

The landmark model and pixel-based model achieved similar standalone performance, but they were not making identical predictions. Their disagreement created an opportunity for aggregation.

### 2. Softmax-style aggregation improved accuracy

The combined model added the probability outputs from the landmark and pixel models and renormalized them into final probabilities. This improved reported test accuracy from approximately 58% to 76.7%.

### 3. Confidence/entropy helped explain reliability

The manuscript found that lower-entropy predictions were generally more reliable. This suggests that model confidence can help identify predictions that may require review.

### 4. Landmark features are interpretable but limited

Landmark features capture facial geometry, such as distances and angles between facial points. They are more interpretable than raw pixels, but they miss texture, lighting, and other visual information.

### 5. Pixel models capture richer visual signals

CNN-based models can use structure, texture, and pixel-level patterns, but they are less interpretable and require external checkpoints and more complex dependencies.

---

## My Contribution

This repository is a reimplementation and cleanup of prior collaborative research work. My contribution focused on turning the original research-style code into a more organized, reviewable, and reproducible data science project.

Specifically, I contributed:

- Wrote the full New York Post scraping pipeline, including link extraction, article parsing, metadata collection, and image downloading.
- Built the landmark-based modeling pipeline, including data loading, preprocessing structure, feature-file handling, classifier training, model comparison, prediction export, and saved-model utilities.
- Reimplemented the landmark workflow around engineered facial features, excluding the original landmark-coordinate calculation logic itself.
- Fixed runtime bugs across the codebase, including NYTimes scraper errors, mismatched function arguments, inconsistent date handling, data type mismatches, and exception-handling issues.
- Rewrote and modularized the pixel-based model pipeline, including model loading, image preprocessing, face detection flow, and inference orchestration.
- Rewrote the softmax aggregation logic used to combine landmark and pixel-model probability outputs.
- Cleaned up README commands, file names, project structure, and external dependency documentation.
- Added support for sample data and smoke tests so reviewers can validate the project structure without downloading the full AffectNet dataset or external model checkpoints.

Prior collaborative work contributed to the original research framing, manuscript, model direction, and landmark-coordinate calculation approach.

---

## Repository Structure

```text
image-emotion-classification/
├── README.md
├── requirements/
│   ├── base.txt
│   ├── landmark.txt
│   ├── pixel.txt
│   ├── scraping.txt
│   ├── text.txt
│   ├── dev.txt
│   └── all.txt
├── sample_data/
│   ├── landmark_features_sample.csv
│   ├── xgb_predictions_sample.csv
│   └── bregnext_predictions_sample.csv
├── tests/
│   ├── test_softmax_aggregation.py
│   ├── test_sample_data_schema.py
│   └── test_imports.py
├── data/                         # External dataset files; not included
├── outputs/                      # Generated predictions/evaluation outputs
├── models/                       # Generated trained models
├── scripts/
│   ├── nypost/
│   │   ├── scraper_utils.py
│   │   ├── main.py
│   │   ├── scraper_articles.py
│   │   ├── scraper_images.py
│   │   └── scraper_links.py
│   ├── nytimes/
│   │   ├── article_fetcher.py
│   │   ├── config.py
│   │   ├── main.py
│   │   ├── nyt_api.py
│   │   └── rate_limiter.py
│   ├── landmark_based/
│   │   ├── run_xgb_landmark.py
│   │   ├── data_loader.py
│   │   ├── feature_engineering.py
│   │   ├── image_processing.py
│   │   ├── landmark_utils.py
│   │   ├── main.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── utils.py
│   ├── pixel_based/
│   │   ├── bregnext_model_loader.py
│   │   ├── data_loader.py
│   │   ├── emotion_models_loader.py
│   │   ├── main.py
│   │   ├── model_runner.py
│   │   └── transforms.py
│   ├── text_emotions_classifier.py
│   └── softmax_agg_predictions.py
└── manuscript/
    └── Finding_Emotions_in_Faces_A_Meta_Classifier.pdf
```

---

## Pipeline Overview

```text
Article/Image Collection
        ↓
Text Emotion Model          Face Detection
        ↓                         ↓
Text Emotion Scores          Pixel CNN + Landmark XGBoost
        ↓                         ↓
                Softmax Aggregation
                         ↓
               Final Emotion Prediction
```

### 1. Data collection

Optional scraping scripts collect articles, article metadata, and images from supported news sources.

### 2. Landmark pipeline

The landmark pipeline uses facial landmark coordinates to engineer geometry-based features, including distances and angles between facial points. These features are used to train classical ML classifiers, including XGBoost.

### 3. Pixel-based pipeline

The pixel pipeline uses cropped face images and CNN-based models to estimate emotion probabilities directly from image pixels.

### 4. Text-emotion pipeline

The text pipeline applies a pretrained transformer/VAD-style model to article text and maps text signals into emotion categories.

### 5. Aggregation

The aggregation step combines model probability outputs and produces a final emotion prediction.

---

## Requirements

Dependencies are split by pipeline so users can install only what they need.

### Base utilities

```bash
pip install -r requirements/base.txt
```

### Landmark pipeline

```bash
pip install -r requirements/landmark.txt
```

### Scraping pipeline

```bash
pip install -r requirements/scraping.txt
```

### Text-emotion pipeline

```bash
pip install -r requirements/text.txt
```

### Development and smoke tests

```bash
pip install -r requirements/dev.txt
pytest -q
```

### Pixel-based pipeline

The pixel-based pipeline depends on external checkpoints and may require a separate environment because it uses TensorFlow compatibility-mode graph loading, PyTorch models, MTCNN, and custom model definitions.

```bash
pip install -r requirements/pixel.txt
```

Alternatively:

```bash
conda env create -f requirements/pixel_environment.yml
conda activate emotion-pixel
```

---

## External Files Required

The following files are not included in this repository:

- AffectNet dataset
- BReG-NeXt TensorFlow checkpoint
- AffectNet PyTorch checkpoint
- SongFan checkpoint and model definition
- Large generated output files

These files are excluded because of dataset licensing, storage size, and checkpoint distribution constraints.

---

## Data Availability

The original AffectNet dataset is not included due to size and licensing restrictions. Access should be requested from the official dataset provider.

Generated outputs are also excluded from the repository. Once the required data and checkpoints are available, the scripts can be rerun to regenerate model outputs.

---

## How to Use

### 1. Landmark-based emotion classification

Train and compare landmark-based classifiers:

```bash
python scripts/landmark_preprocessing/main.py \
  --data_dir data \
  --feature_files new_features.csv
```

Run XGBoost prediction on new feature data:

```bash
python scripts/landmark_preprocessing/run_xgb_landmark.py \
  --model_path models/best_model_XGBoost_Crafted_Features.joblib \
  --data_path data/new_features.csv \
  --output_path outputs/xgb_predictions.csv
```

### 2. Pixel-based emotion classification

Run pixel-based inference:

```bash
python scripts/pixel_based/main.py \
  --image_dir data/images \
  --bregnext_ckpt checkpoints/bregnext \
  --affectnet_ckpt checkpoints/affectnet.pth \
  --songfan_ckpt checkpoints/songfan.pth \
  --example_dir examples
```

### 3. Text-based emotion classification

Classify article text into emotion categories:

```bash
python scripts/text_emotions_classifier.py \
  --input_csv data/articles.csv \
  --output_csv outputs/articles_with_emotions.csv \
  --text_column article_text \
  --name_column article_name \
  --batch_size 16
```

### 4. Article scraping

Scrape New York Post articles:

```bash
python scripts/nypost/main.py \
  --year 2023 \
  --month 5 \
  --output_dir data/nypost
```

Scrape New York Times articles:

```bash
python scripts/nytimes/main.py \
  --start_date 2022-01-01 \
  --end_date 2022-03-01 \
  --output_dir outputs/nyt
```

For NYTimes scraping, set the API key as an environment variable:

```bash
export NYT_API_KEY="your_api_key_here"
```

### 5. Aggregate predictions

Combine XGBoost and deep-learning probability outputs:

```bash
python scripts/softmax_agg_predictions.py \
  --xgb_input outputs/xgb_predictions.csv \
  --dl_input outputs/bregnext_predictions.csv \
  --output_csv outputs/combined_predictions.csv
```

---

## Run with Sample Data

This repository includes small synthetic CSV files in `sample_data/` so reviewers can test the project structure without downloading AffectNet or external model checkpoints.

### Run landmark training with sample data

```bash
python scripts/landmark_preprocessing/main.py \
  --data_dir sample_data \
  --feature_files landmark_features_sample.csv
```

### Run softmax aggregation with sample data

```bash
python scripts/softmax_agg_predictions.py \
  --xgb_input sample_data/xgb_predictions_sample.csv \
  --dl_input sample_data/bregnext_predictions_sample.csv \
  --output_csv outputs/combined_sample_predictions.csv
```

Sample CSVs are synthetic and used only for smoke testing. Reported model results come from the full AffectNet-based experiment described in the manuscript.

---

## Smoke Tests

This repository includes lightweight smoke tests to verify:

- sample data schemas
- prediction aggregation logic
- key module imports

Run:

```bash
pytest -q
```

These tests are not full model-validation tests. They are intended to confirm that the repository structure and sample pipeline are functional.

---

## Output Files

Typical generated outputs include:

```text
outputs/
├── xgb_predictions.csv
├── bregnext_predictions.csv
├── combined_predictions.csv
├── articles_with_emotions.csv
└── evaluation/
    ├── classification_report.csv
    ├── confusion_matrix.png
    └── model_comparison.csv
```

### Prediction CSV schema

Prediction files should use probability columns with the `prob_` prefix:

```text
prob_Neutral
prob_Happy
prob_Sad
prob_Surprise
prob_Fear
prob_Disgust
prob_Anger
prob_Contempt
```

This keeps aggregation scripts stable and avoids relying on column positions.

---

## Evaluation Notes

The original experiment used a balanced AffectNet subset and reported:

- landmark model accuracy and F1
- pixel model accuracy
- combined meta-classifier accuracy and F1
- confusion matrices by emotion class
- entropy/confidence analysis
- Cohen’s kappa comparing agreement between landmark and pixel models

The current repository is designed to organize and reproduce this workflow when external data and checkpoints are available.

---

## Limitations

- AffectNet is not included due to licensing and storage restrictions.
- Model checkpoints are not included.
- The pixel-based pipeline depends on legacy TensorFlow compatibility and external model definitions.
- Text-emotion scoring is exploratory and may use heuristic mapping from VAD-style scores to emotion categories.
- Facial emotion recognition estimates visible expression signals, not true internal emotional state.
- The reported results come from the accompanying manuscript and require full-data reproduction.
- Cross-dataset validation is not included.
- Web scraping can break if source-page HTML changes.
- Errors in face detection or landmark extraction can degrade downstream predictions.

---

## Ethical Use

Emotion recognition should be used carefully. This project should not be used for:

- high-stakes decisions about individuals
- medical or psychological diagnosis
- surveillance or identity-based profiling
- claims about a person’s true internal feelings

The intended use is educational and analytical: comparing modeling approaches and exploring aggregate emotional signals in media content.

---

## Key Learnings

- Similar standalone model accuracy does not mean models make the same errors.
- Combining models can improve performance when their errors are complementary.
- Engineered facial geometry features are interpretable but limited.
- Pixel-based CNN models capture richer signals but are harder to reproduce and explain.
- Confidence/entropy can help identify less reliable predictions.
- Reproducibility requires clear data assumptions, external-file documentation, sample data, and tests.

---

## Acknowledgement

This project builds on prior collaborative work. I acknowledge guidance and contributions from:

- Dr. Siddhartha Dalal
- Dr. Michael Lesk
- Wesley Yuan
- Vikki Sui

---

## License

This project is for research and educational purposes.
