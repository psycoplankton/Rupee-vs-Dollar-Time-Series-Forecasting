import pandas as pd
import matplotlib.pyplot as plt
from typing import DataFrame


def remove_nan(dataframe : DataFrame):
    for i in range(len(dataframe)):
        if dataframe['DEXINUS'][i] != '.':  
            dataframe['DEXINUS'][i] = float(dataframe['DEXINUS'][i])
        else:
            dataframe['DEXINUS'][i] = 0
    dataframe['DATE'][i] = pd.to_datetime(dataframe['DATE'][i])

    row_drop = []
    for i in range(len(dataframe)):
        if dataframe['DEXINUS'][i] == 0:
            row_drop.append(i)
        dataframe.drop(row_drop, inplace=True)
        
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
