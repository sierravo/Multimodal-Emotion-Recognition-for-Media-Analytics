import os

API_KEY = os.getenv("NYT_API_KEY")
BASE_URL_TEMPLATE = "https://api.nytimes.com/svc/archive/v1/{year}/{month}.json?api-key={api_key}"
MAX_QUERIES = 4000
QUERY_DELAY = 6  # seconds
