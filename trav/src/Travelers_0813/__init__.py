import os
import pickle
import sys
import pandas as pd
import numpy as np
import torch

from pathlib import Path
import json
from json import JSONEncoder
import csv
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split

class data_setup:
    def __init__(self, df:pd.DataFrame):
        self.df = df
    
    def stratified_train_val_test(self,  stratify_col='convert_ind', test_size=0.2, val_size=0.1, random_state=24):
        # Step 1: Split into train+val and test
        train_val_set, test_set = train_test_split(
            self.df,
            test_size=test_size,
            stratify=self.df[stratify_col],
            random_state=random_state
        )

        # Step 2: Split train_val into train and validation
        # Adjust validation size relative to train_val_set
        val_relative_size = val_size / (1 - test_size)
        train_set, val_set = train_test_split(
            train_val_set,
            test_size=val_relative_size,
            stratify=train_val_set[stratify_col],
            random_state=random_state
        )
        return train_set, val_set, test_set

    def stratified_train_test(self, stratify_col='convert_ind', test_size=0.2, val_size=0.1, random_state=24):  
        # from sklearn.model_selection import train_test_split
        train_set, test_set = train_test_split(self.df, test_size=0.2, stratify=self.df[stratify_col], random_state=24)

        # Separate features and target from the entire training set
        train_y = train_set[stratify_col]
        train_x = train_set.drop(columns=[stratify_col])

        test_y = test_set[stratify_col]
        test_x = test_set.drop(columns=[stratify_col])

        return train_y, train_x, test_y, test_x
    
    

class Feature_engineering:
    def __init__(self, df:pd.DataFrame):
        self.df = df
 
    def fill_mode(self,col_name):
        tmp = self.df[col_name].mode()
        self.df[col_name] = self.df[col_name].fillna(tmp[0])
    
    def fill_mean(self, col_name):
        tmp = self.df[col_name].mean()
        self.df[col_name] = self.df[col_name].fillna(tmp)

    def fill_median(self, col_name:str):
        tmp = self.df[col_name].median()
        self.df[col_name] = self.df[col_name].fillna(tmp)
    

class EDA_tools:
    def __init__(self, df:pd.DataFrame):
        self.df = df

    def see_kurtosis(self, col_name:str):
        print(f'Kurtosis of {col_name}: {self.df[col_name].kurtosis()}')

    def see_skewness(self, col_name:str):
        print(f'Skewness of {col_name}: {self.df[col_name].skew()}')