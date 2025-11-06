import os
import csv
import logging
import argparse
import datetime
from tqdm import tqdm
from dateutil.relativedelta import relativedelta

from nyt_api import get_archive_data, filter_articles
from article_fetcher import fetch_article_text, save_text, fetch_article_media, save_images

def parse_args():
    """
    Parse command line arguments for start and end dates.

    Returns:
        argparse.Namespace: Parsed arguments with 'start' and 'end' dates.
    """
    parser = argparse.ArgumentParser(description='NYT Archive Article & Media Scraper')
    parser.add_argument('--start', type=str, required=False, default='2019-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=False, default='2019-01-01', help='End date (YYYY-MM-DD)')
    return parser.parse_args()

def run_scraper():
    """
    Run the scraper over the specified date range, saving articles, media, and metadata.
    """
    args = parse_args()
    start_date = datetime.datetime.strptime(args.start, '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime(args.end, '%Y-%m-%d').date()

    n_queries = 0
    current_date = start_date

    while current_date <= end_date:
        month, year = current_date.month, current_date.year
        subdir = f"NYT_{month:02}_{year}"
        text_dir = os.path.join('text', subdir)
        media_dir = os.path.join('media', subdir)
        os.makedirs(text_dir, exist_ok=True)
        os.makedirs(media_dir, exist_ok=True)

        metadata_dir = 'metadata'
        os.makedirs(metadata_dir, exist_ok=True)
        metadata_file = os.path.join(metadata_dir, f"{subdir}_media_metadata.csv")

        with open(metadata_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['headline', 'path', 'caption'])

            try:
                df, n_queries = get_archive_data(month, year, n_queries)
                filtered_df = filter_articles(df)
                filtered_df.to_csv(os.path.join(metadata_dir, f"{subdir}_article_metadata.csv"), index=False)

                for i, url in enumerate(tqdm(filtered_df['web_url'], desc=f"Processing {subdir}")):
                    headline, article, n_queries = fetch_article_text(url, n_queries)
                    if not article:
                        continue
                    text_path = os.path.join(text_dir, f"article_{i}.txt")
                    save_text(article, text_path)

                    media, captions, n_queries = fetch_article_media(url, n_queries)
                    if media:
                        img_label = f"{subdir}/article_{i}__NUM_.jpg"
                        img_paths = save_images(media, img_label)
                        for path, caption in zip(img_paths, captions):
                            writer.writerow([headline, path, caption])

            except Exception as e:
                logging.error(f"Failed for {month}/{year}: {e}")

        current_date += relativedelta(months=1)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    run_scraper()
