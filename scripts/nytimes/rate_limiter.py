import time

from config import MAX_QUERIES, QUERY_DELAY


def rate_limit(current_queries, max_queries=MAX_QUERIES, query_delay=QUERY_DELAY):
    """Sleep between API/web requests and enforce a max query count."""
    if current_queries >= max_queries:
        raise RuntimeError("Maximum query limit reached.")
    time.sleep(query_delay)
    return current_queries
