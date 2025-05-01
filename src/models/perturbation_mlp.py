import torch
from torch import nn

#NOTE: This code mostly went unused in our project (it was experimented with briefly). 
# See perturbation_model.py for more project relevant code

class PeturbationMLP(nn.Module):
    """
    An MLP that takes a waveform (B, 1, T)
    and outputs a residual waveform of the same shape.
    Created for our black-box attack on Whisper.

    Args:
        input_size (int): The size of the input layer
        num_layers (int): Number of hidden layers
        hidden_size (int): Width of hidden layers
        max_delta (float): Maximum percentage change to be applied to the waveform
    """

    def __init__(self,
                 input_size: int,
                 num_layers: int,
                 hidden_size: int,
                 max_delta: float):
        super().__init__
        layers: list[nn.Module] = []
        entry_layer = nn.Linear(
            in_features = input_size,
            out_features = hidden_size,
        )
        layers.append(entry_layer)
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_layers):
            hidden_layer = nn.Linear(
                in_features = hidden_size,
                out_features = hidden_size,
            )
            layers.append(hidden_layer)
            layers.append(nn.ReLU(inplace=True))

        exit_layer = nn.Linear(
            in_features = hidden_size,
            out_features = input_size,
        )
        layers.append(exit_layer)
        self.model = nn.Sequential(*layers)
        self.max_delta = max_delta

    def forward(self, in_waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            in_waveform (pt.Tensor): The input waveform of shape (B, 1, T) or (B, T)
        Returns:
            pt.Tensor: The output residual waveform of shape (B, 1, T) or (B, T)
        """
        if len(in_waveform.shape) == 2:
            in_waveform = in_waveform.unsqueeze(1)
        out_waveform = self.model(in_waveform)
        out_waveform = torch.tanh(out_waveform) * self.max_delta
        return out_waveform