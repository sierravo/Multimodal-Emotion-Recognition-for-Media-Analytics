import time
from config import MAX_QUERIES, QUERY_DELAY

def rate_limit(current_count: int, buffer: int = 1) -> int:
    """
    Enforce query rate limits by sleeping and checking maximum allowed queries.

    Args:
        current_count (int): Current number of queries made.
        buffer (int, optional): Number of queries planned for the next request. Defaults to 1.

    Raises:
        RuntimeError: If max queries would be exceeded.

    Returns:
        int: Updated number of queries after this request.
    """
    if current_count + buffer > MAX_QUERIES:
        raise RuntimeError('Maximum queries per day reached.')
    time.sleep(QUERY_DELAY)
    return current_count + buffer
