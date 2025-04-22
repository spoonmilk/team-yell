import torch as pt
import torch.nn as nn


class ClosedPerturbationModel(pt.nn.Module):
    """
    A 2D CNN that takes a log mel spectrogram (B, 1, F, T)
    and outputs a residual log-mel spectrogram of the same shape.
    Created for our black-box attack on Whisper.

    Args:
        band_num (int): The number of mel frequency bands.
        kernel_size (tuple[int, int]): The size of the convolutional kernel.
        num_channels (int): The number of channels in the hidden layers.
        num_layers (int): The number of convolutional layers.
        db_max (float): The maximum dB of perturbation to be applied.
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
        # ReLU activation
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_layers):
            # Convolutional layer
            conv_layer = nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=kernel_size,
                padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            )
            layers.append(conv_layer)
            # ReLU activation
            layers.append(nn.ReLU(inplace=True))

        # Exit layer to convert (B, num_channels, F, T) to (B, 1, F, T)
        exit_layer = nn.Conv2d(
            in_channels=num_channels,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
        )
        layers.append(exit_layer)
        # ReLU activation
        layers.append(nn.ReLU(inplace=True))

        self.model = nn.Sequential(*layers)
        self.db_max = db_max

    def forward(self, in_spectrogram: pt.Tensor) -> pt.Tensor:
        """
        Forward pass of the model.

        Args:
            in_spectrogram (pt.Tensor): The input log mel spectrogram of shape (B, 1, F, T) or (B, F, T).

        Returns:
            pt.Tensor: The output residual log mel spectrogram of shape (B, F, T) clamped to +- db_max.
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

        # Squeeze and clamp output
        x = x.tanh() * self.db_max
        return x.squeeze(1) if in_spectrogram.ndim == 3 else x
