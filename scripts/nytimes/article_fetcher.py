import os
import requests
import logging
from bs4 import BeautifulSoup
from rate_limiter import rate_limit

def fetch_article_text(url: str, current_queries: int):
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
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        body = soup.find('section', {'name': 'articleBody'})
        if not body:
            return None, None, current_queries
        article = ' '.join(p.get_text() for p in body.find_all('p'))
        # Normalize quotes
        article = article.replace(u'‘', "'").replace(u'’', "'").replace(u'“', '"').replace(u'”', '"')
        headline = soup.find('h1', {'data-test-id': 'headline'}).text
        return headline, article, current_queries
    except Exception as e:
        logging.warning(f"Failed to fetch article text from {url}: {e}")
        return None, None, current_queries

def save_text(content: str, path: str):
    """
    Save text content to a file, creating directories if needed.

    Args:
        content (str): Text content to save.
        path (str): File path where to save the text.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def fetch_article_media(url: str, current_queries: int):
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
        response = requests.get(url)
        response.raise_for_status()
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

def save_images(images: list, label_pattern: str) -> list:
    """
    Save image response content to disk with numbered file names.

    Args:
        images (list): List of requests.Response objects with image content.
        label_pattern (str): Pattern for filenames with '_NUM_' as placeholder for numbering.

    Returns:
        list: List of paths where images were saved.
    """
    paths = []
    for i, img in enumerate(images, 1):
        path = os.path.join('media', label_pattern.replace('_NUM_', f"{i:03}"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(img.content)
        paths.append(path)
    return paths
