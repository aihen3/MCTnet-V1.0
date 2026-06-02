import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Channel attention module based on a lightweight MLP.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    reduction : int, default=6
        Reduction ratio for the bottleneck layer.

    Input
    -----
    x : torch.Tensor
        Shape (batch_size, channels)

    Output
    ------
    torch.Tensor
        Shape (batch_size, channels)
    """

    def __init__(self, in_channels, reduction=6):
        super().__init__()

        hidden = max(1, in_channels // reduction)

        self.fc1 = nn.Linear(in_channels, hidden)
        self.fc2 = nn.Linear(hidden, in_channels)

        self.act = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.dim() != 2:
            raise RuntimeError(
                "ChannelAttention expects input shape "
                "(batch_size, channels)."
            )

        out = self.fc2(self.act(self.fc1(x)))

        # Generate channel-wise attention weights
        weights = self.sigmoid(out)

        # Apply attention
        return x * weights


class TransformerConvClassifier(nn.Module):
    """
    Hybrid Transformer-CNN classifier with channel attention.

    Architecture
    ------------
    1. Transformer branch:
       - Linear projection
       - Transformer encoder
       - Global average pooling

    2. Multi-scale CNN branch:
       - Conv1d(kernel=1)
       - Conv1d(kernel=3)
       - Conv1d(kernel=5)
       - Feature concatenation
       - Global average pooling

    3. Feature fusion:
       - Concatenate Transformer and CNN features
       - Channel attention
       - Fully connected classifier

    Parameters
    ----------
    seq_len : int, default=7
        Input sequence length.

    d_model : int, default=14
        Transformer embedding dimension.

    num_heads : int, default=7
        Number of attention heads.

    num_layers : int, default=2
        Number of Transformer encoder layers.

    conv_out_channels : int, default=7
        Output channels for each convolution branch.

    num_classes : int, default=2
        Number of target classes.

    dropout : float, default=0.1
        Transformer dropout rate.
    """

    def __init__(
        self,
        seq_len=7,
        d_model=14,
        num_heads=7,
        num_layers=2,
        conv_out_channels=7,
        num_classes=2,
        dropout=0.1
    ):
        super().__init__()

        assert d_model % num_heads == 0, \
            "d_model must be divisible by num_heads"

        self.seq_len = seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.conv_out_channels = conv_out_channels

        # Project scalar tokens into d_model dimensions
        self.input_proj = nn.Linear(1, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Multi-scale convolution branches
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=conv_out_channels,
            kernel_size=1,
            padding=0
        )

        self.conv2 = nn.Conv1d(
            in_channels=1,
            out_channels=conv_out_channels,
            kernel_size=3,
            padding=1
        )

        self.conv3 = nn.Conv1d(
            in_channels=1,
            out_channels=conv_out_channels,
            kernel_size=5,
            padding=2
        )

        # Total channels after concatenation
        self.conv_total_channels = conv_out_channels * 3

        # Feature dimension after fusion
        combined_dim = d_model + self.conv_total_channels

        # Channel attention
        self.channel_attention = ChannelAttention(
            in_channels=combined_dim,
            reduction=6
        )

        # Classification layer
        self.fc = nn.Linear(combined_dim, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape
            (batch_size, seq_len).

        Returns
        -------
        torch.Tensor
            Classification logits with shape
            (batch_size, num_classes).
        """

        if x.dim() != 2:
            raise RuntimeError(
                f"Expected 2D input "
                f"(batch_size, seq_len), got {x.shape}"
            )

        if x.size(1) != self.seq_len:
            raise RuntimeError(
                f"Expected seq_len={self.seq_len}, "
                f"got {x.size(1)}"
            )

        x = x.float()

        # Transformer branch
        x_seq = x.unsqueeze(-1)
        x_proj = self.input_proj(x_seq)

        tr_out = self.transformer(x_proj)

        # Global average pooling over sequence dimension
        tr_out = tr_out.mean(dim=1)

        # CNN branch
        x_conv = x_seq.transpose(1, 2)

        c1 = self.conv1(x_conv)
        c2 = self.conv2(x_conv)
        c3 = self.conv3(x_conv)

        conv_cat = torch.cat([c1, c2, c3], dim=1)

        # Global average pooling
        conv_feat = conv_cat.mean(dim=2)

        # Feature fusion
        combined = torch.cat(
            [tr_out, conv_feat],
            dim=1
        )

        if combined.dim() != 2:
            raise RuntimeError(
                "Combined feature tensor must be "
                "2-dimensional."
            )

        if combined.size(1) != self.fc.in_features:
            raise RuntimeError(
                f"Feature dimension mismatch: "
                f"{combined.size(1)} != "
                f"{self.fc.in_features}"
            )

        # Channel attention
        attended = self.channel_attention(combined)

        # Classification
        logits = self.fc(attended)

        return logits