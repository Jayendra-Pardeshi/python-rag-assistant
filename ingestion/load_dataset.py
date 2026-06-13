import pandas as pd
import os
from app.config import settings

def load_stackoverflow_data():
    """
    Loads Questions and Answers from CSV files.
    """
    q_path = os.path.join(settings.data_dir, "Questions.csv")
    a_path = os.path.join(settings.data_dir, "Answers.csv")

    if not os.path.exists(q_path) or not os.path.exists(a_path):
        raise FileNotFoundError("CSV files not found in data directory.")

    print("Loading CSV files...")
    df_q = pd.read_csv(q_path)
    df_a = pd.read_csv(a_path)
    
    return df_q, df_a