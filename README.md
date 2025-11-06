# Facial Emotion Recognition using BReG-NeXt and Facial Landmarks

This repository contains code for a facial emotion recognition pipeline that combines deep learning-based feature extraction and classification using BReG-NeXt and facial landmarks based classification using XGBoost. The project uses the AffectNet dataset to train the models.

This repository is a reproduction of a collaborative project originally developed with contributions from myself and a team of contributors. The goal is to replicate and extend the functionality of the original work for further learning, experimentation, and improvement in image-based emotion classification.

## Acknowledgement

I gratefully acknowledge the mentorship and contributions to the original implementation from: 
- Dr. Siddhartha Dalal, Columbia University
- Dr. Michael Lesk, Rutgers University
- Wesley Yuan, Columbia University
- Vikki Sui, Columbia University


---

## Project Structure

```
image-emotion-classification/
├── README.md
├── requirements.txt
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
│   │   ├── data_loader.py             # Load and preprocess landmark data
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

* Python 3.8+
* TensorFlow 1.x (for loading old models)
* OpenCV
* NumPy, Pandas
* scikit-learn
* XGBoost

You can install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Data Availability

The original dataset (e.g., AffectNet) is not included in this repository due to size and licensing restrictions. Please request access from the official source:

👉 [http://mohammadmahoor.com/affectnet/](http://mohammadmahoor.com/affectnet/)

Output results were not stored due to storage constraints. Once the dataset is downloaded and placed into the appropriate folders, you can re-run the pipeline to reproduce the results.

---

## How to Use

1. Facial Landmark Emotion Classification
Use handcrafted features (e.g., inter-point distances, center distances) to classify facial emotions:

```bash
cd scripts/landmark_preprocessing
python main.py
```

To run only the XGBoost classifier:

```bash
python run_xgb_landmark.py
Input: Precomputed .csv files of features
```

Output: CSV of predictions with class probabilities

2. Pixel-Based Emotion Classification
Use BRegNext or other CNN models to classify emotions directly from facial images:

```bash
cd scripts/pixel_based
python main_pipeline.py
```
Input: Folder of cropped face images
Output: CSV with emotion predictions and softmax probabilities

3. Text-Based Emotion Classification
Classify emotions from text content (e.g., from news or dialogue):


4. Web Article Scraping (Optional)
Scrape from New York Post:

```bash
cd scripts/nypost
python scraper_main.py
```

Scrape from New York Times:

```bash
cd scripts/nytimes
python main.py
```
Note: Add your NYTimes API key to config.py

5. Aggregating Predictions
Merge outputs from text, landmark, and image pipelines using a softmax-weighted strategy:

```bash
python scripts/softmax_agg_predictions.py
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

