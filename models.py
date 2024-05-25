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