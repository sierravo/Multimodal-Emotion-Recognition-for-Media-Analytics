import os
from data_loader import load_features
from models import get_classifiers
from train import run_classifiers
from utils import save_model

FEATURES_DIR = "/content/drive/MyDrive/Media_2020-1/facial landmark code/New Landmark Code/affectnet_balanced_processed"
INTER_DIST_PATH = os.path.join(FEATURES_DIR, "inter_dist.csv")
CENTER_DIST_PATH = os.path.join(FEATURES_DIR, "center_dist.csv")
CRAFTED_PATH = os.path.join(FEATURES_DIR, "new_features.csv")

def main():
    # Load data
    df_inter = load_features(INTER_DIST_PATH)
    df_center = load_features(CENTER_DIST_PATH)
    df_crafted = load_features(CRAFTED_PATH)

    classifiers = get_classifiers()

    all_results = []

    print("\n=== Inter Distance Features ===")
    all_results += run_classifiers(df_inter, "Inter Distance", classifiers)

    print("\n=== Center Distance Features ===")
    all_results += run_classifiers(df_center, "Center Distance", classifiers)

    print("\n=== Crafted Features ===")
    all_results += run_classifiers(df_crafted, "Crafted Features", classifiers)

    # Find best model
    best = max(all_results, key=lambda x: x[1])
    best_name, best_acc, best_dataset, best_model = best

    print(f"\nBest Model: {best_name} on {best_dataset} features with Accuracy: {best_acc:.4f}")
    save_model(best_model, best_name, best_dataset)

if __name__ == "__main__":
    main()
