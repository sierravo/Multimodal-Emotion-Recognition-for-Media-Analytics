import time
from config import MAX_QUERIES, QUERY_DELAY

def rate_limit(current_queries, MAX_QUERIES, QUERY_DELAY):
    """
    Enforce query rate limits by sleeping and checking maximum allowed queries by the NY Times.

    Args:
        current_count (int): Current number of queries made.
    Raises:
        RuntimeError: If max queries would be exceeded.

    Returns:
        int: Updated number of queries after this request.
    """
    if current_queries >= MAX_QUERIES:
        raise RuntimeError('Maximum queries per day reached. Sleeping...')
    time.sleep(QUERY_DELAY)
    return current_queries
