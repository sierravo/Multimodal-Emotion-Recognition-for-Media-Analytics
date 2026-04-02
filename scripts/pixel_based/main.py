"""
main.py

Main script to orchestrate image-based emotion recognition using three different models:
- BReG-NeXt (TensorFlow)
- AffectNet AlexNet (PyTorch)
- SongFan EMO6Classifier (PyTorch)

This pipeline loads example images, detects faces, and runs inference across all models.
"""

import argparse
import os

from data_loader import DataLoader
from model_runner import ModelRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pixel-based emotion recognition on a directory of images."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing input .jpg images"
    )
    parser.add_argument(
        "--bregnext_ckpt",
        type=str,
        required=True,
        help="Path to BReG-NeXt checkpoint directory"
    )
    parser.add_argument(
        "--affectnet_ckpt",
        type=str,
        required=True,
        help="Path to AffectNet checkpoint file"
    )
    parser.add_argument(
        "--songfan_ckpt",
        type=str,
        required=True,
        help="Path to SongFan checkpoint file"
    )
    parser.add_argument(
        "--example_dir",
        type=str,
        default="examples",
        help="Directory for saving example outputs"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.image_dir):
        raise ValueError(f"Invalid image directory: {args.image_dir}")

    dl = DataLoader(images_path=args.image_dir, save_examples_dir=args.example_dir)

    runner = ModelRunner(
        bregnext_ckpt=args.bregnext_ckpt,
        affectnet_ckpt=args.affectnet_ckpt,
        songfan_ckpt=args.songfan_ckpt,
        example_dir=args.example_dir,
    )

    for i, img_path in enumerate(dl.images):
        print(f"\nProcessing image {i}: {img_path}")

        new_img = dl.get_new_image(i, img_path=img_path)
        img_faces = dl.find_faces(i, new_img)

        if len(img_faces) == 0:
            print("No faces detected. Skipping.")
            continue

        results = runner.run_all(new_img, img_faces, i)

        print("\nBReG-NeXt:")
        print(results["bregnext"], results["bregnext"].idxmax(axis=1))

        print("\nAffectNet:")
        print(
            results["affectnet"],
            results["affectnet"].idxmax(axis=1),
            results["affectnet_meta"]
        )

        print("\nSongFan:")
        print(
            results["songfan"],
            results["songfan"].idxmax(axis=1),
            results["songfan_meta"]
        )


if __name__ == "__main__":
    main()