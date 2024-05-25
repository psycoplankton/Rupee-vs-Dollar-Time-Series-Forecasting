import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from pytorch_forecasting import TimeSeriesDataSet

        
def plot_graph(dataframe : DataFrame):


    # Specify the desired figure size
    plt.figure(figsize=(15, 8))

    # Plot the time series
    plt.plot(dataframe['DATE'], dataframe['DEXINUS'],linestyle='-', color='blue', linewidth=2, markersize=4)

    # Set plot title and labels
    plt.title('Indian Rupees to U.S. Dollar Spot Exchange Rate')
    plt.xlabel('Date')
    plt.ylabel('Rate')

    # Show the plot
    plt.show()

def plot_train_val_test_splits(train_dataset : TimeSeriesDataSet,
                               val_dataset : TimeSeriesDataSet,
                               test_dataset : TimeSeriesDataSet):
    

    # Plotting the datasets
    plt.figure(figsize=(15, 7))
    plt.title('Training, Validation, and Test Datasets')
    plt.xlabel('Date')
    plt.ylabel('DEXINUS')

    plt.plot(pd.to_datetime(train_dataset['DATE']).to_numpy(), train_dataset['DEXINUS'], label='Training')
    plt.plot(pd.to_datetime(val_dataset['DATE']).to_numpy(), val_dataset['DEXINUS'], label='Validation')
    plt.plot(pd.to_datetime(test_dataset['DATE']).to_numpy(), test_dataset['DEXINUS'], label='Test')

    plt.legend()
    plt.show()

def plot_monthly_average(dataframe : DataFrame):
    #Let's plot the monthy average of the time series data

    #let's create a copy of the dataframe 
    data_1 = dataframe.copy()

    # Convert 'DATE' column to datetime format
    data_1['DATE'] = pd.to_datetime(data_1['DATE'])

    # Set 'DATE' column as the index
    data_1.set_index('DATE', inplace=True)

    # Resample the data to a specific frequency (e.g., monthly) for trend analysis
    monthly_data = data_1.resample('M').mean()

    # Plot the resampled monthly data
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_data.index, monthly_data['DEXINUS'], label='Monthly Average')
    plt.title('Monthly Average Plot')
    plt.xlabel('Date')
    plt.ylabel('Indian Rupees to U.S. Dollar Spot Exchange Rate')
    plt.legend()
    plt.show()

