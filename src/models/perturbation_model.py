import torch as pt
from torch import nn


class WavPerturbationModel(nn.Module):
    """
    1D CNN that takes a waveform (B, 1, T)
    and outputs a residual waveform of the same shape.
    Created for our black-box attack on Whisper

    Args:
        kernel_size (int): The size of the convolutional kernel
        num_channels (int): The number of channels in the hidden layers
        num_layers (int): The number of convolutional layers
        max_delta (float): Maximum percentage change to be applied to the waveform
    """

    def __init__(
        self,
        kernel_size: int,
        num_channels: int,
        num_layers: int,
        max_delta: float,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        entry_layer = nn.Conv1d(
            in_channels=1,
            out_channels=num_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        layers.append(entry_layer)
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_layers):
            conv_layer = nn.Conv1d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            layers.append(conv_layer)
            layers.append(nn.ReLU(inplace=True))

        exit_layer = nn.Conv1d(
            in_channels=num_channels,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        layers.append(exit_layer)
        self.model = nn.Sequential(*layers)
        self.max_delta = max_delta
        self.options = (
            kernel_size,
            num_channels,
            num_layers,
            max_delta,
        )  # Added this to make model copying easier

    def forward(self, in_waveform: pt.Tensor) -> pt.Tensor:
        """
        Args:
            in_waveform (pt.Tensor): The input waveform of shape (B, 1, T) or (B, T)

        Returns:
            pt.Tensor: The output residual waveform of shape (B, T) clamped to +- max_delta
        """
        # Check if the input is 2D or 3D
        if in_waveform.ndim == 2:
            in_waveform = in_waveform.unsqueeze(1)
            x = in_waveform
        elif in_waveform.ndim == 3:
            x = in_waveform
        else:
            raise ValueError(
                f"Input waveform must be 2D or 3D, but got {in_waveform.ndim}D."
            )

        x = self.model(x)
        # Clamp to +- max_delta
        x = x.tanh() * self.max_delta
        return x.squeeze(1) if in_waveform.ndim == 3 else x
