import logging

import pandas as pd

from config import API_KEY, BASE_URL_TEMPLATE
from rate_limiter import rate_limit


def get_archive_data(session, year, month, current_queries):
    """Request NYT archive data for a given month/year."""
    if not API_KEY:
        raise ValueError("Missing NYT_API_KEY environment variable")

    url = BASE_URL_TEMPLATE.format(year=year, month=month, api_key=API_KEY)
    current_queries = rate_limit(current_queries)

    logging.info(f"Requesting archive data for {month}/{year}...")
    response = session.get(url, timeout=20)
    response.raise_for_status()
    current_queries += 1

    data = response.json()
    docs = data.get("response", {}).get("docs", [])
    df = pd.DataFrame(docs)
    if df.empty:
        return df, current_queries

    df["pub_date"] = pd.to_datetime(df["pub_date"], errors="coerce").dt.date
    df["main_headline"] = df["headline"].apply(
        lambda h: h.get("main", "").lower() if isinstance(h, dict) else ""
    )
    df["list_keywords"] = df["keywords"].apply(
        lambda kws: [kw.get("value", "").lower() for kw in kws] if isinstance(kws, list) else []
    )
    return df, current_queries


def filter_articles(df: pd.DataFrame) -> pd.DataFrame:
    """Filter NYT archive metadata to news articles in selected sections."""
    if df.empty:
        return df

    required_cols = ["type_of_material", "print_section", "section_name"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required NYT metadata columns: {missing}")

    is_news = df["type_of_material"] == "News"
    has_print = df["print_section"].notna()
    in_sections = df["section_name"].isin(["World", "U.S."])
    return df[is_news & has_print & in_sections].reset_index(drop=True)
