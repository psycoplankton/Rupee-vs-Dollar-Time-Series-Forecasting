import torch
from torch.utils.data import Dataset, Subset
import numpy as np
from torch.utils.data import DataLoader
from pytorch_forecasting import TimeSeriesDataSet
import pandas as pd
from pandas import DataFrame

def remove_nan(dataframe : DataFrame):
    for i in range(len(dataframe)):
        if dataframe['DEXINUS'][i] != '.':  
            dataframe['DEXINUS'][i] = float(dataframe['DEXINUS'][i])
        else:
            dataframe['DEXINUS'][i] = float(dataframe['DEXINUS'][i-1])

class TrainTestValidationSplit(Dataset):
    def get_train_test_val_splits(self, dataframe : DataFrame, train_ratio : int =0.8, val_ratio : int =0.1, test_ratio : int =0.1):

        #convert the dataframe object to pytorch tensor    
        dataset = TimeSeriesDataSet(data=dataframe,
                                    time_idx='time_idx',
                                    target=['DEXINUS'],
                                    group_ids=['group_ids'])


        #calculate the length of the dataset
        length = len(dataframe)
        train_size = int(train_ratio * length)
        val_size = int(val_ratio * length)
        test_size = length - train_size - val_size

        train_dataset = dataframe[:train_size]
        val_dataset = dataframe[train_size:train_size+val_size]
        test_dataset = dataframe[train_size+val_size:]

        return train_dataset, val_dataset, test_dataset
    
    def get_train_test_val_dataloaders(self, train_dataset : TimeSeriesDataSet, val_dataset : TimeSeriesDataSet, test_dataset : TimeSeriesDataSet, batch_size : int):

        train_dataset = TimeSeriesDataSet(train_dataset, time_idx='time_idx', target='DEXINUS', group_ids=['group_ids'])
        val_dataset = TimeSeriesDataSet(val_dataset, time_idx='time_idx', target='DEXINUS', group_ids=['group_ids'])
        test_dataset = TimeSeriesDataSet(test_dataset, time_idx='time_idx', target='DEXINUS', group_ids=['group_ids'])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader