#Define the N-BEATS, Nhits and Autoformer Architecture
import torch.nn as nn
from pytorch_forecasting.models.nbeats import NBeats
from pytorch_forecasting.models.nhits import NHiTS
from transformers import AutoformerModel, AutoformerConfig
from pytorch_forecasting.metrics.point import MAE, MAPE, SMAPE, MASE, RMSE
import config

class NBEATS():
    def __init__(self,
        stack_types,
        num_blocks,
        num_block_layers,
        widths,
        sharing,
        expansion_coefficient_lengths,
        prediction_length,
        loss,
        logging_metrics         
    ) -> None:
        
        self.stack_types = stack_types,
        self.num_blockes = num_blocks,
        self.num_block_layers = num_block_layers,
        self.widths = widths,
        self.sharing = sharing,
        self.expansion_coefficient_lengths = expansion_coefficient_lengths,
        self.prediction_length = prediction_length,
        self.loss = loss,
        self.logging_metrics = logging_metrics

        
    nbeats = NBeats(
    stack_types = ["generic", "generic", "generic"],
    num_blocks = [3, 3, 3],
    num_block_layers = [4, 4, 4],
    widths = [512, 512, 512],
    sharing = False,
    expansion_coefficient_lengths = [32, 32, 32],
    prediction_length = config.HORIZON,
    loss = MAE(),
    logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()])
)

NHITS = NHiTS(
    output_size = config.HORIZON,
    static_hidden_size = ['512'],
    loss = MAE(),
    logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()]),
    downsample_frequencies = [1, 1, 1],
    pooling_sizes = [1, 1, 1]

)

configuration = AutoformerConfig(
    prediction_length = config.HORIZON,
    loss = 'mae',
    input_size = config.WINDOW_SIZE,
)

autoformer = AutoformerModel(configuration)

import torch
from torch.nn import LSTM, Linear, BatchNorm1d, Dropout
from typing import Sequence
from torch.autograd import Variable

class LSTMModel(torch.nn.Module):
    def __init__(self, input_size : int, hidden_size : int, num_layers : int, output_size : int, dropout = 0.1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.fc_input = Linear(input_size, hidden_size)
        self.lstm = LSTM(hidden_size*4, 512, num_layers, batch_first=True)
        self.fc_output = Linear(512, output_size)
        self.bn = BatchNorm1d(hidden_size)
        self.dropout = Dropout(dropout)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)

        # Initialize hidden state with zeros
        h_0 = Variable(torch.zeros(self.num_layers, batch_size, self.lstm.hidden_size).to(x.device))
        c_0 = Variable(torch.zeros(self.num_layers, batch_size, self.lstm.hidden_size).to(x.device))

        # Process each feature vector independently
        processed_features = []
        for i in range(x.size(1)):  # Iterate over the 4 feature vectors
            feature = x[:, i, :]  # Shape: [batch_size, sequence_length]
            feature = self.fc_input(feature)  # Shape: [batch_size, hidden_size]
            feature = self.bn(feature)  # Apply BatchNorm1d to [batch_size, hidden_size]
            feature = self.dropout(feature)  # Apply dropout
            processed_features.append(feature)

        # Stack processed features and reshape to be compatible with LSTM input
        x = torch.stack(processed_features, dim=1)  # Shape: [batch_size, num_features, hidden_size]
        x = x.view(batch_size, -1, self.hidden_size * 4)  # Combine features into the input size for LSTM

        # LSTM expects input of shape [batch_size, seq_len, input_size]
        x, _ = self.lstm(x, (h_0, c_0))

        # Pass through the output layer
        x = self.fc_output(x[:, -1, :])  # Use the last time step's output
        return x


model = LSTMModel(input_size=config.max_input_length, hidden_size=128, num_layers=3, output_size=config.max_prediction_length)
model.to(device)


