import pandas as pd
from pathlib import Path

def load_parquet_data(file_path: str) -> pd.DataFrame:
    """
    Loads a parquet file into a Pandas DataFrame.
    
    Args:
        file_path (str): Path to the .parquet file.
        
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found at {file_path}")
    
    return pd.read_parquet(path) 