import os

API_KEY = os.getenv("NYT_API_KEY")
if not API_KEY:
    raise ValueError("Missing NYT_API_KEY environment variable")

BASE_URL_TEMPLATE = f"https://api.nytimes.com/svc/archive/v1/{{year}}/{{month}}.json?api-key={API_KEY}"
MAX_QUERIES = 4000
QUERY_DELAY = 6  # seconds delay between queries to respect API rate limits
