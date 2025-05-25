"""
File Utilities

This module contains utility functions for file I/O operations including
saving and loading models, data, and results.
"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Union


class FileUtils:
    """Utility class for file operations"""
    
    @staticmethod
    def save_pickle(data: Any, filename: str, create_dir: bool = True) -> None:
        """
        Save data to pickle file
        
        Args:
            data: Data to save
            filename (str): Target filename
            create_dir (bool): Whether to create directory if it doesn't exist
        """
        if create_dir:
            directory = os.path.dirname(filename)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to {filename}")
    
    @staticmethod
    def load_pickle(filename: str) -> Any:
        """
        Load data from pickle file
        
        Args:
            filename (str): Source filename
            
        Returns:
            Loaded data
        """
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"Data loaded from {filename}")
        return data
    
    @staticmethod
    def save_excel(data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                   filename: str, create_dir: bool = True) -> None:
        """
        Save DataFrame(s) to Excel file
        
        Args:
            data: DataFrame or dict of DataFrames
            filename (str): Target filename
            create_dir (bool): Whether to create directory if it doesn't exist
        """
        if create_dir:
            directory = os.path.dirname(filename)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
        
        if isinstance(data, dict):
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                for sheet_name, df in data.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=True)
        else:
            data.to_excel(filename, index=True)
        
        print(f"Data saved to {filename}")
    
    @staticmethod
    def load_excel(filename: str, sheet_name: Union[str, int, None] = None, 
                   index_col: Union[int, str, None] = 0) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Load DataFrame(s) from Excel file
        
        Args:
            filename (str): Source filename
            sheet_name: Sheet name or number (None for all sheets)
            index_col: Column to use as index
            
        Returns:
            DataFrame or dict of DataFrames
        """
        if sheet_name is None:
            # Load all sheets
            data = pd.read_excel(filename, sheet_name=None, index_col=index_col)
        else:
            # Load specific sheet
            data = pd.read_excel(filename, sheet_name=sheet_name, index_col=index_col)
        
        print(f"Data loaded from {filename}")
        return data
    
    @staticmethod
    def save_csv(data: pd.DataFrame, filename: str, create_dir: bool = True) -> None:
        """
        Save DataFrame to CSV file
        
        Args:
            data (pd.DataFrame): Data to save
            filename (str): Target filename
            create_dir (bool): Whether to create directory if it doesn't exist
        """
        if create_dir:
            directory = os.path.dirname(filename)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
        
        data.to_csv(filename, index=True)
        print(f"Data saved to {filename}")
    
    @staticmethod
    def load_csv(filename: str, index_col: Union[int, str, None] = 0) -> pd.DataFrame:
        """
        Load DataFrame from CSV file
        
        Args:
            filename (str): Source filename
            index_col: Column to use as index
            
        Returns:
            pd.DataFrame: Loaded data
        """
        data = pd.read_csv(filename, index_col=index_col)
        print(f"Data loaded from {filename}")
        return data
    
    @staticmethod
    def create_directory(directory: str) -> None:
        """
        Create directory if it doesn't exist
        
        Args:
            directory (str): Directory path to create
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory created: {directory}")
        else:
            print(f"Directory already exists: {directory}")
    
    @staticmethod
    def list_files(directory: str, extension: str = None) -> List[str]:
        """
        List files in directory
        
        Args:
            directory (str): Directory path
            extension (str): File extension filter (e.g., '.pkl', '.xlsx')
            
        Returns:
            List[str]: List of filenames
        """
        if not os.path.exists(directory):
            print(f"Directory does not exist: {directory}")
            return []
        
        files = os.listdir(directory)
        
        if extension:
            files = [f for f in files if f.endswith(extension)]
        
        return files
    
    @staticmethod
    def file_exists(filename: str) -> bool:
        """
        Check if file exists
        
        Args:
            filename (str): File path to check
            
        Returns:
            bool: True if file exists
        """
        return os.path.exists(filename)
    
    @staticmethod
    def get_file_size(filename: str) -> int:
        """
        Get file size in bytes
        
        Args:
            filename (str): File path
            
        Returns:
            int: File size in bytes
        """
        if os.path.exists(filename):
            return os.path.getsize(filename)
        else:
            return 0
    
    @staticmethod
    def save_results_summary(results: Dict[str, Any], filename: str) -> None:
        """
        Save results summary to text file
        
        Args:
            results (dict): Results dictionary
            filename (str): Target filename
        """
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(filename, 'w') as f:
            f.write("ABCM Model Results Summary\n")
            f.write("=" * 50 + "\n\n")
            
            for key, value in results.items():
                f.write(f"{key}:\n")
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        f.write(f"  {sub_key}: {sub_value}\n")
                else:
                    f.write(f"  {value}\n")
                f.write("\n")
        
        print(f"Results summary saved to {filename}")
    
    @staticmethod
    def backup_file(filename: str, backup_suffix: str = '.bak') -> str:
        """
        Create backup of existing file
        
        Args:
            filename (str): Original filename
            backup_suffix (str): Suffix for backup file
            
        Returns:
            str: Backup filename
        """
        if os.path.exists(filename):
            backup_filename = filename + backup_suffix
            
            # If backup already exists, add number
            counter = 1
            while os.path.exists(backup_filename):
                backup_filename = f"{filename}.{counter}{backup_suffix}"
                counter += 1
            
            # Copy file
            import shutil
            shutil.copy2(filename, backup_filename)
            print(f"Backup created: {backup_filename}")
            return backup_filename
        else:
            print(f"Original file does not exist: {filename}")
            return "" 