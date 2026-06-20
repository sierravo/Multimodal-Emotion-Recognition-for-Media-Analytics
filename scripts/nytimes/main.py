import argparse
import os
from datetime import datetime

import pandas as pd
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from article_fetcher import create_session, fetch_article_media, fetch_article_text, save_images, save_text
from nyt_api import filter_articles, get_archive_data


def parse_args():
    parser = argparse.ArgumentParser(description="NYT Archive Article & Media Scraper")
    parser.add_argument("--start_date", type=str, default="2019-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2019-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output_dir", type=str, default="data/nyt_articles", help="Base output directory")
    return parser.parse_args()


def ensure_dirs(base_dir):
    text_dir = os.path.join(base_dir, "text")
    media_dir = os.path.join(base_dir, "media")
    metadata_dir = os.path.join(base_dir, "metadata")
    for path in [text_dir, media_dir, metadata_dir]:
        os.makedirs(path, exist_ok=True)
    return text_dir, media_dir, metadata_dir


def main():
    args = parse_args()
    session = create_session()
    current_queries = 0
    text_dir, media_dir, metadata_dir = ensure_dirs(args.output_dir)
    output_csv_path = os.path.join(metadata_dir, "nyt_articles.csv")

    start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.strptime(args.end_date, "%Y-%m-%d")

    all_results = []
    current = start.replace(day=1)
    while current <= end:
        year, month = current.year, current.month
        print(f"\nFetching archive for {year}-{month:02d}")
        archive_df, current_queries = get_archive_data(session, year, month, current_queries)
        filtered_df = filter_articles(archive_df)

        for idx, article in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Processing NYT articles"):
            url = article.get("web_url")
            if not url:
                continue
            try:
                headline, text, current_queries = fetch_article_text(session, url, current_queries)
                if not text:
                    continue

                image_urls, captions, current_queries = fetch_article_media(session, url, current_queries)
                file_stem = f"nyt_{year}_{month:02d}_{idx}"
                text_path = save_text(text, headline, file_stem, text_dir)
                image_paths = save_images(session, image_urls, file_stem, media_dir)

                all_results.append({
                    "article_name": headline,
                    "article_text": text,
                    "text_path": text_path,
                    "image_urls": image_urls,
                    "image_paths": image_paths,
                    "captions": captions,
                    "url": url,
                    "year": year,
                    "month": month,
                })
                print(f"Processed: {headline}")
            except Exception as e:
                print(f"[!] Error processing {url}: {e}")
                continue

        current += relativedelta(months=1)

    df = pd.DataFrame(all_results)
    df.to_csv(output_csv_path, index=False)
    print(f"\nSaved {len(df)} articles to {output_csv_path}")


if __name__ == "__main__":
    main()
