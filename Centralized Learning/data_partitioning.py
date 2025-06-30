import pandas as pd
import numpy as np
import os
import argparse
from sklearn.utils import shuffle


def split_dataset(input_file, output_dir, ratios, random_state=23):
    """
    Splits a dataset into multiple client datasets based on given ratios.
    
    Args:
        input_file (str): Path to the input CSV file
        output_dir (str): Directory to save client datasets
        ratios (list): List of ratios for splitting 
        random_state (int): Random state for shuffling
    
    Returns:
        list: Paths of created client CSV files
    """
    dataset = pd.read_csv(input_file)
    dataset = shuffle(dataset, random_state=random_state)
    
    total_ratio = sum(ratios)
    n_total = len(dataset)
    client_sizes = [int((r / total_ratio) * n_total) for r in ratios]
    
    client_sizes[-1] = n_total - sum(client_sizes[:-1])
    
    os.makedirs(output_dir, exist_ok=True)
    
    created_files = []
    start = 0
    
    for i, size in enumerate(client_sizes):
        end = start + size
        client_df = dataset.iloc[start:end].reset_index(drop=True)
        
        csv_path = os.path.join(output_dir, f"client_{i+1}.csv")
        client_df.to_csv(csv_path, index=False)
        created_files.append(csv_path)
        
        start = end
    
    return created_files


def main():
    parser = argparse.ArgumentParser(description='Split dataset into client datasets')
    parser.add_argument('--input_file', help='Path to input CSV file')
    parser.add_argument('--output_dir', help='Directory to save client datasets')
    parser.add_argument('--ratios', nargs='+', type=int, default=[5, 3, 2],
                        help='Ratios for splitting dataset (default: 5 3 2)')
    parser.add_argument('--random-state', type=int, default=23,
                        help='Random state for shuffling (default: 23)')
    
    args = parser.parse_args()
    
    try:
        created_files = split_dataset(
            args.input_file, 
            args.output_dir, 
            args.ratios, 
            args.random_state
        )
        
        print(f"Successfully created {len(created_files)} client datasets:")
        for file_path in created_files:
            print(f"  - {file_path}")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())