from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "landmark_based"))
sys.path.insert(0, str(ROOT / "scripts" / "nypost"))
sys.path.insert(0, str(ROOT / "scripts" / "nytimes"))


def test_core_imports():
    import softmax_agg_predictions  # noqa: F401
    # text_emotions_classifier requires optional Hugging Face dependencies.
    # It is tested by syntax compilation in the smoke suite, not imported here.
    import landmark_utils  # noqa: F401
    import data_loader  # noqa: F401


def test_scraper_imports():
    import scraper_utils  # noqa: F401
    import scraper_links  # noqa: F401
    import scraper_articles  # noqa: F401
    import scraper_images  # noqa: F401


def test_nyt_imports_without_api_key_error():
    import config  # noqa: F401
    import rate_limiter  # noqa: F401
