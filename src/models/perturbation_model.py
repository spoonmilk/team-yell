import torch as pt
from torch import nn


class SpectroPerturbationModel(nn.Module):
    """
    A 2D CNN that takes a log mel spectrogram (B, 1, F, T)
    and outputs a residual log-mel spectrogram of the same shape.
    Created for our black-box attack on Whisper

    Args:
        band_num (int): The number of mel frequency bands
        kernel_size (tuple[int, int]): The size of the convolutional kernel
        num_channels (int): The number of channels in the hidden layers
        num_layers (int): The number of convolutional layers
        db_max (float): The maximum dB of perturbation to be applied
    """

    def __init__(
        self,
        kernel_size: tuple[int, int],
        num_channels: int,
        num_layers: int,
        db_max: float,
    ):
        # Entry layer to convert (B, 1, F, T) to (B, num_channels, F, T)
        layers: list[nn.Module] = []
        entry_layer = nn.Conv2d(
            in_channels=1,
            out_channels=num_channels,
            kernel_size=kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
        )
        layers.append(entry_layer)
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_layers):
            conv_layer = nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=kernel_size,
                padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            )
            layers.append(conv_layer)
            layers.append(nn.ReLU(inplace=True))

        # Exit layer to convert (B, num_channels, F, T) to (B, 1, F, T)
        exit_layer = nn.Conv2d(
            in_channels=num_channels,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
        )
        layers.append(exit_layer)
        layers.append(nn.ReLU(inplace=True))

        self.model = nn.Sequential(*layers)
        self.db_max = db_max

    def forward(self, in_spectrogram: pt.Tensor) -> pt.Tensor:
        """
        Args:
            in_spectrogram (pt.Tensor): The input log mel spectrogram of shape (B, 1, F, T) or (B, F, T)

        Returns:
            pt.Tensor: The output residual log mel spectrogram of shape (B, F, T) clamped to +- db_max
        """
        # Check if the input is 3D or 4D
        if in_spectrogram.ndim == 3:
            in_spectrogram = in_spectrogram.unsqueeze(1)
            x = in_spectrogram
        elif in_spectrogram.ndim == 4:
            x = in_spectrogram
        else:
            raise ValueError(
                f"Input spectrogram must be 3D or 4D, but got {in_spectrogram.ndim}D."
            )

        # Pass through model
        x = self.model(x)
        # Clamp to +- db_max
        x = x.tanh() * self.db_max
        return x.squeeze(1) if in_spectrogram.ndim == 3 else x


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
        layers.append(nn.ReLU(inplace=True))
        self.model = nn.Sequential(*layers)
        self.max_delta = max_delta
        self.options = (kernel_size, num_channels, num_layers, max_delta)  # Added this to make model copying easier

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
        return x.squeeze(1) if in_waveform.ndim == 2 else x
