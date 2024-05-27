import torch
from torch.utils.data import Dataset, Subset
import numpy as np
from torch.utils.data import DataLoader
from pytorch_forecasting import TimeSeriesDataSet
import pandas as pd
from pandas import DataFrame
import sklearn
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List

def remove_nan(dataframe : DataFrame):
    for i in range(len(dataframe)):
        if dataframe['DEXINUS'][i] != '.':  
            dataframe['DEXINUS'][i] = float(dataframe['DEXINUS'][i])
        else:
            dataframe['DEXINUS'][i] = float(dataframe['DEXINUS'][i-1])


class TrainTestSplit():
    def get_train_test_splits(self, dataframe : pd.DataFrame, dataset_class : Dataset, train_ratio : float =0.8):

        #calculate the length of the dataset
        split = int(train_ratio * len(dataframe))

        train_dataframe = dataframe[:split]
        test_dataframe = dataframe[split:]

        train_dataset = dataset_class(
            data=train_dataframe,
            group_ids = ['group_ids'],
            time_idx = 'time_idx',
            target = 'DEXINUS',
            max_encoder_length = max_input_length,
            min_encoder_length = max_input_length,
            max_prediction_length = max_prediction_length,
            time_varying_unknown_reals = ['DEXINUS']
        )

        test_dataset = train_dataset.from_dataset(
            dataset = train_dataset,
            data = df,
            min_prediction_idx = split,
        )
        return train_dataset, test_dataset


    def get_train_test_dataloaders(self, train_dataset, test_dataset, batch_size):

        train_loader = train_dataset.to_dataloader(train = True, batch_size = batch_size, num_workers=2)

        test_loader = test_dataset.to_dataloader(train = False, batch_size = batch_size, num_workers=2)

        return train_loader, test_loader
    


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
        year = torch.Tensor(le.fit_transform(df['Year'].to_numpy())) #(1, 3132)

        year = torch.reshape(year, (year.shape[0], 1)) # (3132, 1)
        pad = config.max_input_length - year.shape[1]
        year = torch.nn.functional.pad(year, (0, pad)) # (3132, 60)
        #emb = torch.randn((Year_sine.shape[1], max_input_length), generator=g)
        #year_embeddings = Year_sine @ emb # (13, 60)
        return torch.tensor(year)

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
        Month = cyc.fit_transform(torch.tensor(df['Month'].to_numpy()), period=12) #2, 3132
        Month = torch.reshape(Month, (Month.shape[1], Month.shape[0]))
        pad = config.max_input_length - Month.shape[1]
        Month = torch.nn.functional.pad(Month, (0, pad))
        #emb = torch.randn((Month.shape[1], max_input_length), generator=g)
        #month_embeddings = Month @ emb # (12, 60)
        return Month

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
        Day = cyc.fit_transform(torch.tensor(df['Day'].to_numpy()), period=31) #2, 3132
        Day = torch.reshape(Day, (Day.shape[1], Day.shape[0])) # (3132, 2)
        pad = config.max_input_length - Day.shape[1]
        Day = torch.nn.functional.pad(Day, (0, pad))
        #emb = torch.randn((Day.shape[1], max_input_length), generator=g)
        #day_embeddings = Day @ emb # (31, 60)
        return Day

    def train_test_split(self, dataframe : DataFrame):
        df = dataframe.copy()
        df.drop(['Year', 'Month', 'Day', 'DATE'], axis= 1, inplace=True)
        labels , inputs = [], []
        labels.append('DEXINUS')
        for i in range(config.max_prediction_length-1):
            labels.append(f'DEXINUS_shift_{i+1}')
        for i in range(config.max_input_length):
            inputs.append(f'DEXINUS_shift_{i + 30}')

        df_inputs = df.drop(columns = labels)
        df_labels = df.drop(columns = inputs)

        inputs = torch.Tensor(df_inputs.to_numpy(dtype=float))
        labels = torch.Tensor(df_labels.to_numpy(dtype=float))

        return inputs, labels

    def labeldict(self, labels : torch.Tensor, values : torch.Tensor) -> Dict[float, int]:
        return {labels[i]:values[i] for i in range(len(labels))}

    def get_dataloaders(self, inputs : torch.Tensor, labels : torch.Tensor, embeddings : List[torch.Tensor], dataset_class : TimeSeriesDataset, ratio : float = 0.8) -> DataLoader:
        features = torch.stack((inputs,
                                embeddings[0],
                                embeddings[1],
                                embeddings[2]), dim=1)


        split = int(0.8 * inputs.shape[0])
        train_inputs, train_labels = features[:split], labels[:split]
        test_inputs, test_labels = features[split:], labels[split:]
        print(train_inputs.shape), print(test_inputs.shape)
        training_data = dataset_class(train_inputs, train_labels)
        test_data = dataset_class(test_inputs, test_labels)

        train_loader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

        return train_loader, test_loader