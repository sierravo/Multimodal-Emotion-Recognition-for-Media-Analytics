import pandas as pd

def load_features(path):
    """
    Load feature CSV file into a pandas DataFrame.
    """
    return pd.read_csv(path)
