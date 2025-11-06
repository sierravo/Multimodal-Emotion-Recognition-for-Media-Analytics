from scraper_utils import ensure_directory, build_subdir, get_day_range
from scraper_links import get_nypost_links
from scraper_articles import extract_article_data
from scraper_images import save_images

import os
import csv
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

MAX_THREADS = 10

def scrape_nypost_articles(year: str, month: str) -> None:
    """Scrapes NYPost articles for a given month and year and saves the data and images."""
    ensure_directory("media")
    ensure_directory("metadata")

    all_links = []
    print(f"Collecting links for {year}-{month}...")
    for day in tqdm(range(1, get_day_range(year, month) + 1), desc="Fetching daily article links"):
        date_str = f"{day:02}"
        links = get_nypost_links(year, month, date_str)
        all_links.extend(links)

    print(f"Extracting {len(all_links)} articles...")
    articles = []
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {executor.submit(extract_article_data, url): url for url in all_links}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Parsing articles"):
            data = future.result()
            if data['section_tag'] == "News" and "No image" not in data['image_links']:
                articles.append(data)

    if not articles:
        print("[!] No valid articles found.")
        return

    df = pd.DataFrame(articles)
    subdir = build_subdir('metadata', month, year)
    ensure_directory(subdir)

    df.to_csv(f'{subdir}/data.csv', index=False)
    print("Article data saved.")

    def download_images(i, row):
        label = f'article_{i}_NUM.jpg'
        return [(row['headline'], path) for path in save_images(row['image_links'], month, year, label)]

    with open(f'{subdir}/media_metadata.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['headline', 'path'])

        print("Downloading images...")
        image_jobs = []
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            futures = [executor.submit(download_images, i, row) for i, row in df.iterrows()]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading images"):
                image_jobs.extend(future.result())

        for headline, path in image_jobs:
            writer.writerow([headline, path])

    print("Done!")
