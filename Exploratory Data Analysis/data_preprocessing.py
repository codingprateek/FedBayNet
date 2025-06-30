import pandas as pd
import numpy as np
import re
import argparse
import os

class DataPreprocessor:
    """Performs data preprocessing, including encoding of ordinal and binary variables."""
    
    def __init__(self, input_path: str, output_path: str = None):
        self.input_path = input_path
        self.output_path = output_path or self._default_output_path()
        self.data = None

    def default_output_path(self):
        base_name = os.path.splitext(os.path.basename(self.input_path))[0]
        output_dir = os.path.dirname(self.input_path) or "."
        return os.path.join(output_dir, f"encoded_{base_name}.csv")
        
    def load_data(self) -> pd.DataFrame:
        self.data = pd.read_csv(self.input_path)
        self.data = self.data.iloc[2:].reset_index(drop=True)
        print(f"Data loaded: {self.data.shape}")
        return self.data
    
    def preprocess(self) -> pd.DataFrame:
        self.load_data()
        self.encode_range_variables()
        self.encode_binary_variables()
        self.handle_special_cases()
        self.clean_column_names()
        self.drop_redundant_columns()
        return self.data
    
    def save_data(self):
        if self.data is not None:
            self.data.to_csv(self.output_path, index=False)
            print(f"Data saved to: {self.output_path}")
    
    def encode_binary_variables(self):
        binary_vars = [
            'bp (Diastolic)', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'class'
        ]
        for var in binary_vars:
            if var in self.data.columns:
                self.encode_binary_column(var)
    
    def encode_binary_column(self, column):
        mapping = {
            '0': 0, '1': 1,
            'ckd': 1, 'notckd': 0
        }
        self.data[column] = self.data[column].astype(str).str.strip().str.lower()
        self.data[column] = self.data[column].map(mapping)
    
    def encode_range_variables(self):
        binary_vars = [
            'bp (Diastolic)', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'class'
        ]
        range_vars = [col for col in self.data.columns if col not in binary_vars]
        for var in range_vars:
            self.encode_range_column(var)
    
    def encode_range_column(self, column):
        def get_sorted_values(val):
            val = val.strip()
            if re.match(r'^<\s*\d+', val):
                return float(val.replace('<', '').strip()) - 1e-5  
            elif re.match(r'^≥\s*\d+', val):
                return float(val.replace('≥', '').strip()) + 1e-5  
            elif re.match(r'^\d+', val):
                parts = re.findall(r'\d+(?:\.\d+)?', val)
                if len(parts) == 2:
                    return (float(parts[0]) + float(parts[1])) / 2 
            return float('inf')  

        unique_values = self.data[column].dropna().unique()
        sorted_values = sorted(unique_values, key=get_sorted_values)
        range_mapping = {val: idx for idx, val in enumerate(sorted_values)}
        self.data[column] = self.data[column].map(range_mapping)
    
    def handle_special_cases(self):
        if 'grf' in self.data.columns:
            self.data['grf'] = self.data['grf'].replace(10, np.nan)
            mode_value = self.data['grf'].mode().iloc[0] if not self.data['grf'].mode().empty else 0
            self.data['grf'] = self.data['grf'].fillna(mode_value)
            self.data['grf'] = self.data['grf'].astype(int)

    def clean_column_names(self):
        rename_dict = {
            'bp (Diastolic)': 'bp_diastolic',
            'bp limit': 'bp_limit'
        }
        self.data = self.data.rename(columns=rename_dict)

    def drop_redundant_columns(self):
        if 'affected' in self.data.columns:
            self.data = self.data.drop(columns=['affected'])

