import os
import calendar

def ensure_directory(path: str) -> str:
    """Create directory if it doesn't exist and return the path."""
    os.makedirs(path, exist_ok=True)
    return path

def build_subdir(base: str, month: str, year: str) -> str:
    """Generate a standardized subdirectory path for storing data."""
    return f'{base}/NYPost_{month}_{year}'

def get_day_range(year: str, month: str) -> int:
    """Return the number of days in the specified month and year."""
    return calendar.monthrange(int(year), int(month))[1]
