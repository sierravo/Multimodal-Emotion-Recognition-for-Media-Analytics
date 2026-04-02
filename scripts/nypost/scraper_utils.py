import os
import calendar
import requests
import argparse

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

TIMEOUT = 20


def get_session():
    session = requests.Session()
    session.headers.update(HEADERS)
    return session

def parse_args():
    parser = argparse.ArgumentParser(description="NY Post scraper")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/nypost",
        help="Base output directory"
    )
    return parser.parse_args()

def build_dirs(base_dir, year, month):
    subdir = f"NYPost_{month}_{year}"

    media_dir = os.path.join(base_dir, "media", subdir)
    metadata_dir = os.path.join(base_dir, "metadata", subdir)

    os.makedirs(media_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    return media_dir, metadata_dir


def ensure_directory(path: str) -> str:
    """Create directory if it doesn't exist and return the path."""
    os.makedirs(path, exist_ok=True)
    return path




def get_day_range(year: str, month: str) -> int:
    """Return the number of days in the specified month and year."""
    return calendar.monthrange(int(year), int(month))[1]
