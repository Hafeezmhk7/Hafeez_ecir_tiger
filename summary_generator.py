"""
Script to download product images from a DataFrame containing image URLs
and generate a summary column for product descriptions.
"""

import os
import requests
from tqdm import tqdm
import pandas as pd
import argparse
from modules.utils import display_args
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from transformers import pipeline
import torch

def df_stats(df: pd.DataFrame, title="DataFrame Stats"):
    table = Table(title=title)
    rprint(f"DataFrame shape: {df.shape}")
    table.add_column("Column", style="cyan", no_wrap=True)
    table.add_column("Non-Null Count", style="yellow")
    table.add_column("Unique Count", style="magenta")
    table.add_column("Null/NA Count", style="red")
    table.add_column("Data Type", style="green")

    for col in df.columns:
        try:
            non_null_count = df[col].notna().sum()
        except:
            non_null_count = "Error"
        try:
            unique_count = df[col].nunique(dropna=True)
        except:
            unique_count = "Error"
        try:
            null_count = df[col].isna().sum()
        except:
            null_count = "Error"
        try:
            dtype = str(df[col].dtype)
        except:
            dtype = "Error"
        table.add_row(col, str(non_null_count), str(unique_count), str(null_count), dtype)

    Console().print(table)
    

def generate_description_summary(df):
    summarizer = pipeline("summarization", model="knkarthick/MEETING_SUMMARY", device=0 if torch.cuda.is_available() else -1)

    texts = df["description"].fillna("").astype(str).tolist()
    summaries = []
    batch_size = 256
    for i in tqdm(range(0, len(texts), batch_size), desc="Summarizing descriptions"):
        batch = texts[i:i+batch_size]
        # truncate each item in batch
        batch = [t[:1024] if len(t) > 1024 else t for t in batch]
        try:
            batch_summaries = summarizer(batch, max_length=32, min_length=8, do_sample=False)
            summaries.extend([x['summary_text'] for x in batch_summaries])
        except Exception:
            # fallback: add empty summary for failed batch
            summaries.extend([""] * len(batch))

    df["description_summary"] = summaries
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize product descriptions.")
    parser.add_argument('--dataset_dir', type=str, default="./dataset/amazon/raw", help='Directory containing the dataset')
    parser.add_argument('--dataset_split', type=str, default="beauty", help='Dataset split to process')
    args = parser.parse_args()
    display_args(args)

    # use provided arguments or default values
    DATASET_DIR = args.dataset_dir
    DATASET_SPLIT = args.dataset_split

    df_path = f"{DATASET_DIR}/{DATASET_SPLIT}/item_data.csv"
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"DataFrame file {df_path} does not exist.")
    
    df = pd.read_csv(df_path)
    df_stats(df, title=f"Dataset: {DATASET_SPLIT} - {df_path}")

    # generate summaries
    df = generate_description_summary(df)

    # save with summaries
    df.to_csv(df_path, index=False)
    rprint(f"[green]Updated DataFrame with 'description_summary' saved to {df_path}[/green]")