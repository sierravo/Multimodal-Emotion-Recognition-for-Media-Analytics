API_KEY = 'GF0YptMX0CePLe9qqfY2WYAPNyGCGj7n'
BASE_URL_TEMPLATE = 'https://api.nytimes.com/svc/archive/v1/{year}/{month}.json?api-key=' + API_KEY
MAX_QUERIES = 4000
QUERY_DELAY = 6  # seconds delay between queries to respect API rate limits
