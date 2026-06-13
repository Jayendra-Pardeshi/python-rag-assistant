import pandas as pd
from bs4 import BeautifulSoup
import re

def clean_html(text):
    if not isinstance(text, str):
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ")

def merge_and_clean(df_q, df_a, sample_size=5000):
    """
    Merges Questions with Answers and cleans the text.
    """
    print("Merging and cleaning data...")
    
    # Merge Questions and Answers
    df_merged = df_q.merge(df_a, left_on='Id', right_on='ParentId', how='inner')
    
    # Filter high quality
    df_merged = df_merged[df_merged['Score_y'] > 0]
    
    # Sample
    df_merged = df_merged.sample(n=min(sample_size, len(df_merged)), random_state=42)
    
    # Combine Title + Body for better context
    df_merged['Full_Question'] = (
        df_merged['Title'] + " " + df_merged['Body_x'].apply(clean_html)
    )
    
    # Clean Answers
    df_merged['Clean_Answer'] = df_merged['Body_y'].apply(clean_html)
    
    # Prepare text pairs for embedding
    # Format: "Question: ... \n Answer: ..."
    texts = df_merged.apply(
        lambda row: f"Question: {row['Full_Question']}\n\nAnswer: {row['Clean_Answer']}", 
        axis=1
    ).tolist()
    
    metadatas = [
        {"source": "stackoverflow", "id": str(idx)} 
        for idx in df_merged.index
    ]
    
    return texts, metadatas