import argparse
import os

from data_loader import load_features
from models import get_classifiers
from train import run_classifiers
from utils import save_model

DEFAULT_FEATURE_FILES = {
    "Crafted Features": "new_features.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train and compare landmark-based emotion classifiers.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing processed feature CSV files. Defaults to <repo>/data",
    )
    parser.add_argument(
        "--feature_files",
        nargs="*",
        default=None,
        help="Optional feature CSV filenames, e.g. --feature_files new_features.csv",
    )
    parser.add_argument(
        "--quick_smoke",
        action="store_true",
        help="Run only a lightweight classifier for repository smoke testing.",
    )
    return parser.parse_args()


def resolve_repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_data_dir(user_data_dir=None):
    if user_data_dir:
        return user_data_dir
    return os.path.join(resolve_repo_root(), "data")


def build_feature_map(feature_files):
    if feature_files:
        return {
            os.path.splitext(os.path.basename(fname))[0].replace("_", " ").title(): fname
            for fname in feature_files
        }
    return DEFAULT_FEATURE_FILES


def main():
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    feature_map = build_feature_map(args.feature_files)

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            "Create the directory and place processed CSV feature files there."
        )

    classifiers = get_classifiers()
    if args.quick_smoke:
        classifiers = {"Naive_Bayes": classifiers["Naive_Bayes"]}
    all_results = []

    for dataset_name, filename in feature_map.items():
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            print(f"Skipping {dataset_name}: file not found at {file_path}")
            continue

        print(f"\n=== {dataset_name} ===")
        df = load_features(file_path)
        all_results += run_classifiers(df, dataset_name, classifiers)

    if not all_results:
        raise ValueError("No feature files were loaded. Check your data directory and filenames.")

    best_name, best_score, best_dataset, best_model = max(all_results, key=lambda x: x[1])
    print(f"\nBest Model: {best_name} on {best_dataset} with Macro F1: {best_score:.4f}")
    save_model(best_model, best_name, best_dataset)


if __name__ == "__main__":
    main()
