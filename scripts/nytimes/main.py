import os
import csv
import logging
import argparse
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from dateutil.relativedelta import relativedelta

from nyt_api import get_archive_data, filter_articles
from article_fetcher import create_session, fetch_article_text, save_text, fetch_article_media, save_images
from rate_limiter import rate_limit

def parse_args():
    """
    Parse command line arguments for start and end dates.

    Returns:
        argparse.Namespace: Parsed arguments with 'start' and 'end' dates.
    """
    parser = argparse.ArgumentParser(description='NYT Archive Article & Media Scraper')
    parser.add_argument('--start_date', type=str, required=False, default='2019-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, required=False, default='2019-01-01', help='End date (YYYY-MM-DD)')
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/nyt_articles",
        help="Base directory for saving scraped data"
    )

    return parser.parse_args()

def ensure_dirs(base_dir):
    text_dir = os.path.join(base_dir, "text")
    media_dir = os.path.join(base_dir, "media")
    metadata_dir = os.path.join(base_dir, "metadata")

    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(media_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    return text_dir, media_dir, metadata_dir


def main():
    args = parse_args()

    session = create_session()
    current_queries = 0

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Build output file path
    output_csv_path = os.path.join(args.output_dir, "nyt_articles.csv")

    # Parse dates
    start = datetime.datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(args.end_date, "%Y-%m-%d")

    all_results = []

    current = start
    while current <= end:
        year = current.year
        month = current.month

        print(f"\nFetching archive for {year}-{month:02d}")

        archive, current_queries = get_archive_data(
            session,
            year,
            month,
            current_queries
        )

        docs = archive.get("response", {}).get("docs", [])

        for article in docs:
            url = article.get("web_url")
            if not url:
                continue

            try:
                headline, text, current_queries = fetch_article_text(
                    session,
                    url,
                    current_queries
                )

                if not text:
                    continue

                image_urls, current_queries = fetch_article_media(
                    session,
                    url,
                    current_queries
                )

                all_results.append({
                    "article_name": headline,
                    "article_text": text,
                    "image_urls": image_urls,
                    "url": url,
                    "year": year,
                    "month": month
                })

                print(f"Processed: {headline}")

            except Exception as e:
                print(f"[!] Error processing {url}: {e}")
                continue

        # Move to next month
        if month == 12:
            current = datetime.datetime(year + 1, 1, 1)
        else:
            current = datetime.datetime(year, month + 1, 1)

    # Save results


    df = pd.DataFrame(all_results)
    df.to_csv(output_csv_path, index=False)

    print(f"\nSaved {len(df)} articles to {output_csv_path}")


if __name__ == "__main__":
    main()
