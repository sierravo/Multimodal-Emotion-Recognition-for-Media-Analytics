import os
import requests
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from rate_limiter import rate_limit


def create_session():
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    session.headers.update({
        "User-Agent": "Mozilla/5.0"
    })

    return session

def fetch_article_text(session, url, current_queries):
    """
    Fetch the main article text and headline from an NYT article URL.

    Args:
        url (str): Article URL.
        current_queries (int): Current number of queries done.

    Returns:
        tuple: (headline str or None, article text str or None, updated query count)
    """
    current_queries = rate_limit(current_queries)
    try:
        response = session.get(url, timeout = 20)
        response.raise_for_status()

        current_queries += 1

        soup = BeautifulSoup(response.text, 'lxml')
        body = soup.find('section', {'name': 'articleBody'})

        if not body:
            return None, None, current_queries
        article = ' '.join(p.get_text() for p in body.find_all('p'))

        # Normalize quotes
        article = article.replace(u'‘', "'").replace(u'’', "'").replace(u'“', '"').replace(u'”', '"')
        headline_tag = soup.find("h1", {"data-test-id": "headline"})
        headline = headline_tag.get_text(strip=True) if headline_tag else None
        return headline, article, current_queries
    
    except Exception as e:
        logging.warning(f"Failed to fetch article text from {url}: {e}")
        return None, None, current_queries

def save_text(text, headline, filename, text_dir):
    """
    Save text content to a file, creating directories if needed.

    Args:
        content (str): Text content to save.
        path (str): File path where to save the text.
    """
    filepath = os.path.join(text_dir, f"{filename}.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        if headline:
            f.write(headline + "\n\n")
        f.write(text)

def fetch_article_media(session, url, current_queries):
    """
    Fetch media (images) and captions from an NYT article URL.

    Args:
        url (str): Article URL.
        current_queries (int): Current number of queries done.

    Returns:
        tuple: (list of image response objects, list of captions, updated query count)
    """
    current_queries = rate_limit(current_queries)

    try:
        response = session.get(url, timeout = 20)
        response.raise_for_status()

        current_queries += 1

        soup = BeautifulSoup(response.text, 'lxml')
        figures = soup.find_all('figure', {'aria-label': 'media'})
        img_urls, captions = [], []

        for fig in figures:
            try:
                img_url = fig.find('img')['src']
                caption = fig.find('figcaption').text.split('Credit...')[0].strip()
                img_urls.append(img_url)
                captions.append(caption)
            except Exception:
                continue

        imgs = []

        for img_url in img_urls:
            current_queries = rate_limit(current_queries)
            img_resp = requests.get(img_url)
            imgs.append(img_resp)
        return imgs, captions, current_queries
        
    except Exception as e:
        logging.warning(f"Failed to fetch media from {url}: {e}")
        return [], [], current_queries

def save_images(session, image_urls, filename, media_dir):
    """
    Save image response content to disk with numbered file names.

    Returns:
        list: List of paths where images were saved.
    """
    for idx, url in enumerate(image_urls):
        try:
            response = session.get(url, timeout = 20)
            if response.status_code == 200:
                img_path = os.path.join(media_dir, f"{filename}_{idx}.jpg")
                with open(img_path, "wb") as f:
                    f.write(response.content)
        except Exception:
            continue
