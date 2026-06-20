import logging
import os

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import MAX_QUERIES, QUERY_DELAY
from rate_limiter import rate_limit


def create_session():
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    return session


def fetch_article_text(session, url, current_queries):
    current_queries = rate_limit(current_queries, max_queries=MAX_QUERIES, query_delay=QUERY_DELAY)
    try:
        response = session.get(url, timeout=20)
        response.raise_for_status()
        current_queries += 1

        soup = BeautifulSoup(response.text, "lxml")
        body = soup.find("section", {"name": "articleBody"})
        if not body:
            return None, None, current_queries

        article = " ".join(p.get_text() for p in body.find_all("p"))
        article = article.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
        headline_tag = soup.find("h1", {"data-test-id": "headline"})
        headline = headline_tag.get_text(strip=True) if headline_tag else None
        return headline, article, current_queries
    except Exception as e:
        logging.warning(f"Failed to fetch article text from {url}: {e}")
        return None, None, current_queries


def save_text(text, headline, filename, text_dir):
    os.makedirs(text_dir, exist_ok=True)
    filepath = os.path.join(text_dir, f"{filename}.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        if headline:
            f.write(headline + "\n\n")
        f.write(text or "")
    return filepath


def fetch_article_media(session, url, current_queries):
    current_queries = rate_limit(current_queries, max_queries=MAX_QUERIES, query_delay=QUERY_DELAY)
    try:
        response = session.get(url, timeout=20)
        response.raise_for_status()
        current_queries += 1

        soup = BeautifulSoup(response.text, "lxml")
        figures = soup.find_all("figure", {"aria-label": "media"})
        img_urls, captions = [], []
        for fig in figures:
            img = fig.find("img")
            if not img or not img.get("src"):
                continue
            img_urls.append(img["src"])
            caption_tag = fig.find("figcaption")
            captions.append(caption_tag.get_text(strip=True) if caption_tag else "")
        return img_urls, captions, current_queries
    except Exception as e:
        logging.warning(f"Failed to fetch media from {url}: {e}")
        return [], [], current_queries


def save_images(session, image_urls, filename, media_dir):
    os.makedirs(media_dir, exist_ok=True)
    paths = []
    for idx, url in enumerate(image_urls):
        try:
            response = session.get(url, timeout=20)
            response.raise_for_status()
            img_path = os.path.join(media_dir, f"{filename}_{idx}.jpg")
            with open(img_path, "wb") as f:
                f.write(response.content)
            paths.append(img_path)
        except Exception as e:
            logging.warning(f"Failed to save image {url}: {e}")
    return paths
