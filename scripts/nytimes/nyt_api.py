import requests
import pandas as pd
from rate_limiter import rate_limit
from config import BASE_URL_TEMPLATE
import logging


def get_archive_data(session, month, year, current_queries):
    """
    Request archive data for a given month and year from NYT API.

    Args:
        month (int): Month number (1-12).
        year (int): Year number (e.g. 2019).
        current_queries (int): Number of queries already performed.

    Returns:
        tuple: (pandas.DataFrame with article metadata, updated query count)
    """
    url = BASE_URL_TEMPLATE.format(year=year, month=month)
    current_queries = rate_limit(current_queries)

    logging.info(f"Requesting archive data for {month}/{year}...")
    response = session.get(url)
    response.raise_for_status()
    current_queries += 1

    data = response.json()
    docs = data['response']['docs']
    
    df = pd.DataFrame(docs)
    df['pub_date'] = pd.to_datetime(df['pub_date']).dt.date
    df['main_headline'] = df['headline'].apply(lambda h: h.get('main', '').lower())
    df['list_keywords'] = df['keywords'].apply(lambda kws: [kw['value'].lower() for kw in kws])
    return df, current_queries

def filter_articles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter articles to keep only News type, with print section, and specific sections.

    Args:
        df (pd.DataFrame): DataFrame with NYT article metadata.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    is_news = df['type_of_material'] == 'News'
    has_print = df['print_section'].notna()
    in_sections = df['section_name'].isin(['World', 'U.S.'])
    filtered = df[is_news & has_print & in_sections].reset_index(drop=True)
    return filtered
