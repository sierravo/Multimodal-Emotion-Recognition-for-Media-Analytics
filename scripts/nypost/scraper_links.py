import re
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://nypost.com/YEAR/MONTH/DATE/"

def get_nypost_links(session, year: str, month: str, day: str) -> list:
    """Fetch and return a list of article URLs from NYPost for the given date."""
    url = BASE_URL.replace("YEAR", year).replace("MONTH", month).replace("DATE", day)
    try:
        response = session.get(url, timeout = 20)
        response.raise_for_status()
    except Exception as e:
        print(f"[!] Error fetching {url}: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    links = {
        link_tag['href'] for link_tag in soup.find_all('a', href=True)
        if url in link_tag['href'] and re.match(r'https://nypost.com/\d+/\d+/\d+/[-\w]+/?', link_tag['href'])
    }
    
    return list(links)
