import re
import requests
from bs4 import BeautifulSoup

def extract_article_data(session, url):
    try:
        response = session.get(url, timeout = 20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        def safe_find_text(tag, attrs=None, default=None):
            found = soup.find(tag, attrs=attrs) if attrs else soup.find(tag)
            return found.get_text(strip=True) if found else default

        section_tag = safe_find_text('p', {'class': 'section-tag'})
        headline = safe_find_text('h1', default="No headline")
        author_text = safe_find_text('p', {'class': 'byline'})
        if author_text:
            author = author_text.replace("By ", "")
        else:
            author = None
        date_tag = soup.find('p', {'class': 'byline-date'})
        publish_date = re.search(r'\d{4}-\d{2}-\d{2}', str(date_tag)).group(0) if date_tag else "No date"
        publish_time = re.search(r'\d{2}:\d{2}', str(date_tag)).group(0) if date_tag else "No time"

        paragraphs = soup.find_all('div', class_='entry-content entry-content-read-more')
        article_text = ' '.join(p.get_text() for block in paragraphs for p in block.find_all('p')) if paragraphs else "No content"

        images = [img['src'] for div in soup.find_all('div', class_='featured-image') for img in div.find_all('img')]
        captions = []
        for caption_div in soup.find_all("div", class_="wp-caption-text featured"):
            span = caption_div.find("span")
            captions.append(span.text.strip() if span else "No caption")

        return {
            'section_tag': section_tag,
            'headline': headline,
            'article_link': url,
            'author': author,
            'publish_date': publish_date,
            'publish_time': publish_time,
            'text': article_text,
            'image_links': images if images else ["No image"],
            'image_caption': captions if captions else ["No caption"]
        }

    except Exception as e:
        print(f"[!] Failed to parse article: {url}\n    {e}")
        return {
            'section_tag': None,
            'headline': None,
            'article_link': url,
            'author': None,
            'publish_date': None,
            'publish_time': None,
            'text': None,
            'image_links': [None],
            'image_caption': [None]
        }
