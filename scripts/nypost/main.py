from scraper_utils import get_session, ensure_directory, build_dirs, get_day_range
from scraper_links import get_nypost_links
from scraper_articles import extract_article_data
from scraper_images import save_images

import os
import csv
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse

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

MAX_THREADS = 10

def main() -> None:
    """Scrapes NYPost articles for a given month and year and saves the data and images."""
    args = parse_args()
    session = get_session()
    media_dir, metadata_dir = build_dirs(
        args.output_dir,
        args.year,
        args.month
        )

    all_links = []
    print(f"Collecting links for {args.year}-{args.month}...")
    for day in tqdm(range(1, get_day_range(args.year, args.month) + 1), ...):
    links = get_nypost_links(session, args.year, args.month, date_str)

        date_str = f"{day:02}"
        links = get_nypost_links(session, year, month, date_str)
        all_links.extend(links)
        all_links = list(dict.fromkeys(all_links))

    print(f"Extracting {len(all_links)} articles...")
    articles = []
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        future_to_link = {
            executor.submit(extract_article_data, session, url): url 
            for url in all_links
        }    
        for future in tqdm(as_completed(future_to_link), total=len(future_to_link), desc="Parsing articles"):
            url = future_to_link[future]
            try:
                data = future.result()
            except Exception as e:
                print(f"[!] Error in worker: {e}")
                continue            
            if data is not None and data['section_tag'] == "News" and "No image" not in data['image_links']:
                articles.append(data)

    if not articles:
        print("[!] No valid articles found.")
        return

    df = pd.DataFrame(articles)
    csv_path = os.path.join(metadata_dir, "articles.csv")
    df.to_csv(csv_path, index=False)
    print("Article data saved.")

    def download_images(i, row):
        label = f'article_{i}_NUM.jpg'
        return [(row['headline'], path) for path in save_images(session, row['image_links'], label, media_dir)]

    with open(f'{media_dir}/media_metadata.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['headline', 'path'])

        print("Downloading images...")
        image_jobs = []
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            futures = [executor.submit(download_images, i, row) for i, row in df.iterrows()]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading images"):
                try:
                    result = future.result()
                    image_jobs.extend(result)
                except Exception as e:
                    print(f"[!] Error downloading image: {e}")

        for headline, path in image_jobs:
            writer.writerow([headline, path])

    print("Done!")

if __name__ == "__main__":
    main()

