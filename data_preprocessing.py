import torch
from torch.utils.data import Dataset, Subset
import numpy as np
from torch.utils.data import DataLoader
from pytorch_forecasting import TimeSeriesDataSet
import pandas as pd
from pandas import DataFrame
import sklearn
from sklearn.preprocessing import LabelEncoder

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
    
from pandas import DataFrame
import sklearn
from sklearn.preprocessing import LabelEncoder
class PreProcessing():
    def __init__(self):
        pass
    #shift the columns
    def preprocess_dataframe(self, dataframe : DataFrame, window_size : int, ) -> DataFrame:
        """
        Helps create a dataframe with the necessary windows
        Args:
            data: the pandas DataFrame to be shifted
            window_size: the number of time steps to shift
        """
        df = dataframe.copy()
        dt = pd.to_datetime(df['DATE'].to_numpy())
        df['Year'] = dt.year
        df['Month'] = dt.month
        df['Day'] = dt.day
        shitfs = torch.arange(start = 1, end = window_size)
        for i in range(window_size-1):
            df[f'DEXINUS_shift_{i+1}'] = df['DEXINUS'].shift(shitfs[i])
        df.dropna(inplace=True)
        return df

    def encode_year(self, dataframe : DataFrame, transform : sklearn.preprocessing) -> torch.Tensor:
        """
        Encodes the year column
        Args:
            df: the dataframe to be encoded
        """
        df = dataframe.copy()
        g = torch.Generator().manual_seed(2184239752)
        le = transform
        year = torch.Tensor(le.fit_transform(df['Year'].unique())) #(13, )
        period = df['Year'].nunique()
        Year_sine = torch.sin((2 * torch.pi) * year / period) #(13, )
        Year_sine = torch.reshape(Year_sine, (Year_sine.shape[0], 1))
        emb = torch.randn((Year_sine.shape[1], config.max_input_length), generator=g)
        year_embeddings = Year_sine @ emb # (13, 60)
        return year_embeddings

    def encode_month(self, dataframe : DataFrame, transform) -> torch.Tensor:
        """
        Encodes the month column
        Args:
            df: the dataframe to be encoded
            transform: the transform to apply to the month column
        """
        df = dataframe.copy()
        g = torch.Generator().manual_seed(2184239752)
        cyc = transform
        Month = cyc.fit_transoform(torch.tensor(np.sort(df['Month'].unique())), period=12) #2, 12
        Month = torch.reshape(Month, (Month.shape[1], Month.shape[0]))
        emb = torch.randn((Month.shape[1], config.max_input_length), generator=g)
        month_embeddings = Month @ emb # (12, 60)
        return month_embeddings

    def encode_day(self, dataframe : DataFrame, transform) -> torch.Tensor:
        """
        Encodes the day column
        Args:
            df: the dataframe to be encoded
            transform: the transform to apply to the day column
        """
        df = dataframe.copy()
        g = torch.Generator().manual_seed(2184239752)
        cyc = transform
        Day = cyc.fit_transoform(torch.tensor(np.sort(df['Day'].unique())), period=31) #2, 31
        Day = torch.reshape(Day, (Day.shape[1], Day.shape[0]))
        emb = torch.randn((Day.shape[1], config.max_input_length), generator=g)
        day_embeddings = Day @ emb # (31, 60)
        return day_embeddings

    def train_test_split(self, dataframe : DataFrame):
        df = dataframe.copy()
        df.drop(['Year', 'Month', 'Day', 'DATE'], axis= 1, inplace=True)
        test , train = [], []
        test.append('DEXINUS')
        for i in range(config.max_prediction_length-1):
            test.append(f'DEXINUS_shift_{i+1}')
        for i in range(config.max_input_length):
            train.append(f'DEXINUS_shift_{i + 30}')

        df_train = df.drop(columns = test)
        df_test = df.drop(columns = train)

        train_dataset = torch.Tensor(df_train.to_numpy(dtype=float))
        test_dataset = torch.Tensor(df_test.to_numpy(dtype=float))

        return train_dataset, test_dataset