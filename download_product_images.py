"""
Script to download product images from a DataFrame containing image URLs.
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
    

def download_images(df, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    total_images = df['imUrl'].nunique()
    print(f"Total images to download: {total_images}")
    downloaded_images = 0
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Downloading images"):
        url = row['imUrl']
        asin = row['asin']
        if not isinstance(url, str) or not url.startswith('http'):
            continue  # Skip invalid URLs
        try:
            # extract image filename from URL
            filename = asin + ".jpg"
            # optionally, prefix with row index or asin to avoid collisions
            file_path = os.path.join(save_folder, filename)
            
            # skip if file already exists
            if os.path.exists(file_path):
                # print(f'Skipping {url}, file already exists: {file_path}')
                downloaded_images += 1
                continue
            # write the image to the specified path
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(file_path, 'wb') as f:
                f.write(response.content)
            downloaded_images += 1
        except Exception as e:
            print(f'Failed to download {url}: {e}')
            continue

    print(f"Downloaded `{downloaded_images}` out of `{total_images}` images.")

if __name__ == "__main__":
    argparse.ArgumentParser(description="Download product images from URLs in a DataFrame.")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default="./dataset/amazon/raw", help='Directory containing the dataset')
    parser.add_argument('--dataset_split', type=str, default="beauty", help='Dataset split to process')
    args = parser.parse_args()
    display_args(args)

    # use provided arguments or default values
    DATASET_DIR = args.dataset_dir
    DATASET_SPLIT = args.dataset_split
    
    # ensure the dataset directory exists
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Dataset directory {DATASET_DIR} does not exist.")
    if not os.path.exists(f"{DATASET_DIR}/{DATASET_SPLIT}"):
        raise FileNotFoundError(f"Dataset split {DATASET_SPLIT} does not exist in {DATASET_DIR}.")
    # ensure the item_data.csv file exists
    df_path = f"{DATASET_DIR}/{DATASET_SPLIT}/item_data.csv"
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"DataFrame file {df_path} does not exist.")
    
    # load the DataFrame containing image URLs
    df_path = f"{DATASET_DIR}/{DATASET_SPLIT}/item_data.csv"
    df = pd.read_csv(df_path)
    df_stats(df, title=f"Dataset: {DATASET_SPLIT} - {df_path}")
    
    # download images   
    save_path = f"{DATASET_DIR}/{DATASET_SPLIT}/product_images"
    download_images(df, save_path)

