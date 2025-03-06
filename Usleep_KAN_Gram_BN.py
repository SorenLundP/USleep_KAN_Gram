import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import lru_cache
from typing import Tuple, Optional, Union, Callable, Type, List


class BottleneckGRAMLayer(nn.Module):
    """
    Bottleneck version of the GRAM layer that uses dimensionality reduction
    to significantly decrease parameters while maintaining performance.
    
    The architecture follows: 
    1. Dimensionality reduction (squeezing)
    2. GRAM polynomial basis transformation
    3. Dimensionality restoration (expansion)
    4. Residual connection
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        reduction_factor: int = 4, 
        degree: int = 2, 
        act: Type[nn.Module] = nn.SiLU,
        dropout_rate: float = 0.0,
        layer_norm: bool = True
    ):
        """
        Initialize Bottleneck GRAM Layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            reduction_factor: Factor for dimensionality reduction in bottleneck
            degree: Maximum degree of Gram polynomials
            act: Activation function class
            dropout_rate: Dropout probability (0.0 = no dropout)
            layer_norm: Whether to use layer normalization
        """
        super(BottleneckGRAMLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduced_channels = max(1, in_channels // reduction_factor)
        self.degrees = degree
        self.dropout_rate = dropout_rate

        # Dimensionality reduction (squeezing)
        self.squeeze = nn.Linear(in_channels, self.reduced_channels)
        
        # GRAM polynomial transformation components
        self.act = act()
        
        if layer_norm:
            self.norm = nn.LayerNorm(self.reduced_channels, dtype=torch.float32)
        else:
            self.norm = nn.Identity()
            
        self.beta_weights = nn.Parameter(torch.zeros(degree + 1, dtype=torch.float32))
        self.grams_basis_weights = nn.Parameter(
            torch.zeros(self.reduced_channels, self.reduced_channels, degree + 1, dtype=torch.float32)
        )
        self.base_weights = nn.Parameter(
            torch.zeros(self.reduced_channels, self.reduced_channels, dtype=torch.float32)
        )
        
        # Dimensionality expansion
        self.expand = nn.Linear(self.reduced_channels, out_channels)
        
        # Residual connection if input and output dimensions match
        self.has_residual = (in_channels == out_channels)
        if not self.has_residual and in_channels != out_channels:
            self.residual_proj = nn.Linear(in_channels, out_channels)
            
        # Dropout for regularization
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = nn.Identity()

        self.init_weights()

    def init_weights(self):
        """Initialize weights with appropriate scaling."""
        # Initialize beta weights
        nn.init.normal_(
            self.beta_weights,
            mean=0.0,
            std=1.0 / (self.reduced_channels * (self.degrees + 1.0)),
        )

        # Initialize GRAM basis weights
        nn.init.xavier_uniform_(self.grams_basis_weights)
        nn.init.xavier_uniform_(self.base_weights)
        
        # Initialize projection layers
        nn.init.xavier_uniform_(self.squeeze.weight)
        nn.init.zeros_(self.squeeze.bias)
        nn.init.xavier_uniform_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)
        
        # Initialize residual projection if needed
        if hasattr(self, 'residual_proj'):
            nn.init.xavier_uniform_(self.residual_proj.weight)
            nn.init.zeros_(self.residual_proj.bias)

    def beta(self, n: int, m: int) -> torch.Tensor:
        """
        Compute beta coefficient for Gram polynomial recurrence relation.
        
        Args:
            n: Current polynomial degree
            m: Next polynomial degree
            
        Returns:
            Beta coefficient tensor
        """
        return (
            ((m + n) * (m - n) * n ** 2) / (m ** 2 / (4.0 * n ** 2 - 1.0))
        ) * self.beta_weights[n]

    @lru_cache(maxsize=128)
    def gram_poly(self, x: torch.Tensor, degree: int) -> torch.Tensor:
        """
        Compute Gram polynomial basis functions up to specified degree.
        
        Args:
            x: Input tensor
            degree: Maximum polynomial degree
            
        Returns:
            Tensor of Gram polynomial values
        """
        p0 = x.new_ones(x.size())

        if degree == 0:
            return p0.unsqueeze(-1)

        p1 = x
        grams_basis = [p0, p1]

        for i in range(2, degree + 1):
            p2 = x * p1 - self.beta(i - 1, i) * p0
            grams_basis.append(p2)
            p0, p1 = p1, p2

        # Stack along dim=1 to get shape (batch_size * L, d, k)
        grams_basis_stacked = torch.stack(grams_basis, dim=1)
        return grams_basis_stacked

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Bottleneck GRAM layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after transformation
        """
        # Save input for residual connection
        identity = x
        
        # Apply dropout for regularization (noise injection before KAN as per paper)
        x = self.dropout(x)
        
        # Step 1: Dimensionality reduction (squeezing)
        x = self.squeeze(x)
        
        # Step 2: Apply activation
        basis = F.linear(self.act(x), self.base_weights)
        
        # Step 3: Apply non-linear transformation
        x = torch.tanh(x).contiguous()
        
        # Step 4: Generate polynomial features
        grams_basis = self.act(self.gram_poly(x, self.degrees))
        
        # Step 5: Tensor contraction using einsum
        y = torch.einsum(
            "b d k, k o d -> b o",
            grams_basis,
            self.grams_basis_weights,
        )
        
        # Step 6: Normalize and activate
        y = self.act(self.norm(y + basis))
        
        # Step 7: Dimensionality expansion
        y = self.expand(y)
        
        # Step 8: Apply residual connection if shapes match
        if self.has_residual:
            y = y + identity
        elif hasattr(self, 'residual_proj'):
            y = y + self.residual_proj(identity)
            
        return y


class BottleneckGRAMKANConv1d(nn.Module):
    """
    1D Convolutional layer using Bottleneck GRAM transformation to reduce parameters
    while maintaining model expressivity.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int = 1, 
        padding: int = 0, 
        degree: int = 2,
        reduction_factor: int = 4,
        dropout_rate: float = 0.0
    ):
        """
        Initialize Bottleneck GRAM KAN 1D Convolutional layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolving kernel
            stride: Stride of the convolution
            padding: Padding added to both sides of the input
            degree: Maximum degree of Gram polynomials
            reduction_factor: Factor for bottleneck reduction
            dropout_rate: Dropout probability for regularization
        """
        super(BottleneckGRAMKANConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Create bottleneck GRAM layer for the vectorized patches
        self.kanlayer = BottleneckGRAMLayer(
            in_channels=in_channels * kernel_size, 
            out_channels=out_channels,
            reduction_factor=reduction_factor,
            degree=degree,
            dropout_rate=dropout_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Bottleneck GRAM KAN 1D Convolutional layer.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, length)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, output_length)
        """
        batch_size, in_channels, length = x.size()
        assert in_channels == self.in_channels, f"Expected {self.in_channels} input channels, got {in_channels}"

        # Apply unfold to get sliding local blocks
        # Reshape x to (batch_size, in_channels, 1, length) to mimic 2D data with height=1
        x_reshaped = x.unsqueeze(2)  # (batch_size, in_channels, 1, length)
        x_unfold = F.unfold(
            x_reshaped,
            kernel_size=(1, self.kernel_size),
            stride=(1, self.stride),
            padding=(0, self.padding)
        )  # (batch_size, in_channels * kernel_size, L)

        x_unfold = x_unfold.transpose(1, 2)  # (batch_size, L, in_channels * kernel_size)
        x_unfold = x_unfold.reshape(batch_size * x_unfold.size(1), -1)  # (batch_size * L, in_channels * kernel_size)

        # Apply bottleneck GRAM transformation
        out_unfold = self.kanlayer(x_unfold)  # (batch_size * L, out_channels)

        # Reshape and transpose to get the final output
        out_unfold = out_unfold.reshape(batch_size, -1, out_unfold.size(1))  # (batch_size, L, out_channels)
        out = out_unfold.transpose(1, 2)  # (batch_size, out_channels, L)
        
        # Calculate output length
        out_length = (length + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.reshape(batch_size, self.out_channels, out_length)

        return out


class ConvBNELU_BottleneckGRAM(nn.Module):
    """
    Convolutional block with Bottleneck GRAM, BatchNorm, and ELU activation.
    This is a drop-in replacement for the ConvBNELU module in usleep.py.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int = 6, 
        kernel_size: int = 9, 
        dilation: int = 1, 
        ceil_pad: bool = False, 
        degree: int = 2,
        reduction_factor: int = 4,
        dropout_rate: float = 0.0
    ):
        """
        Initialize ConvBNELU block with Bottleneck GRAM convolution.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolving kernel
            dilation: Dilation rate for convolution
            ceil_pad: Whether to pad if dimension is uneven
            degree: Maximum degree of Gram polynomials
            reduction_factor: Factor for bottleneck reduction
            dropout_rate: Dropout probability for regularization
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (
            self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1) - 1
        ) // 2

        self.layers = nn.Sequential(
            nn.ConstantPad1d(
                padding=(self.padding, self.padding), value=0
            ),  # Padding to maintain sequence length
            BottleneckGRAMKANConv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=0,  # Already handled by ConstantPad1d
                degree=degree,
                reduction_factor=reduction_factor,
                dropout_rate=dropout_rate
            ),
            nn.ELU(),
            nn.BatchNorm1d(self.out_channels),
        )
        self.ceil_pad = ceil_pad
        self.ceil_padding = nn.Sequential(nn.ConstantPad1d(padding=(0, 1), value=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ConvBNELU_BottleneckGRAM block.
        
        Args:
            x: Input tensor
            
        Returns:
            Processed tensor after convolution, normalization and activation
        """
        x = self.layers(x)

        # Add padding after since changing decoder kernel from 3 to 2 introduced mismatch
        if (self.ceil_pad) and (x.shape[2] % 2 == 1):  # Pad 1 if dimension is uneven
            x = self.ceil_padding(x)

        return x


class Encoder_BottleneckGRAM(nn.Module):
    """
    Encoder with Bottleneck GRAM convolutions for the USleep architecture.
    """
    def __init__(
        self,
        filters: List[int],
        max_filters: int,
        in_channels: int = 2,
        maxpool_kernel: int = 2,
        kernel_size: int = 9,
        dilation: int = 1,
        degree: int = 1,
        reduction_factor: int = 4,
        dropout_rate: float = 0.0
    ):
        """
        Initialize Encoder with Bottleneck GRAM convolutions.
        
        Args:
            filters: List of filter counts for each encoder block
            max_filters: Maximum number of filters (for bottleneck)
            in_channels: Number of input channels
            maxpool_kernel: Kernel size for max pooling
            kernel_size: Kernel size for convolutions
            dilation: Dilation rate for convolutions
            degree: Maximum degree of Gram polynomials
            reduction_factor: Factor for bottleneck reduction
            dropout_rate: Dropout probability for regularization
        """
        super().__init__()

        self.filters = filters
        self.in_channels = in_channels
        self.maxpool_kernel = maxpool_kernel
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.depth = len(self.filters)

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBNELU_BottleneckGRAM(
                        in_channels=self.in_channels if k == 0 else self.filters[k - 1],
                        out_channels=self.filters[k],
                        kernel_size=self.kernel_size,
                        dilation=self.dilation,
                        ceil_pad=True,
                        degree=degree,
                        reduction_factor=reduction_factor,
                        dropout_rate=dropout_rate
                    )
                )
                for k in range(self.depth)
            ]
        )
        self.maxpools = nn.ModuleList(
            [nn.MaxPool1d(self.maxpool_kernel) for _ in range(self.depth)]
        )
        self.bottom = nn.Sequential(
            ConvBNELU_BottleneckGRAM(
                in_channels=self.filters[-1],
                out_channels=max_filters,
                kernel_size=self.kernel_size,
                degree=degree,
                reduction_factor=reduction_factor,
                dropout_rate=dropout_rate
            )
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass of the encoder.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (encoded features, skip connections)
        """
        shortcuts = []  # Residual connections
        for layer, maxpool in zip(self.blocks, self.maxpools):
            z = layer(x)
            shortcuts.append(z)
            x = maxpool(z)

        encoded = self.bottom(x)

        return encoded, shortcuts


class Decoder_BottleneckGRAM(nn.Module):
    """
    Decoder with Bottleneck GRAM convolutions for the USleep architecture.
    """
    def __init__(
        self,
        filters: List[int],
        max_filters: int,
        upsample_kernel: int = 2,
        kernel_size: int = 9,
        degree: int = 1,
        reduction_factor: int = 4,
        dropout_rate: float = 0.0
    ):
        """
        Initialize Decoder with Bottleneck GRAM convolutions.
        
        Args:
            filters: List of filter counts for each decoder block
            max_filters: Maximum number of filters (from bottleneck)
            upsample_kernel: Kernel size for upsampling
            kernel_size: Kernel size for convolutions
            degree: Maximum degree of Gram polynomials
            reduction_factor: Factor for bottleneck reduction
            dropout_rate: Dropout probability for regularization
        """
        super().__init__()

        self.filters = filters
        self.upsample_kernel = upsample_kernel
        self.in_channels = max_filters
        self.kernel_size = kernel_size

        self.depth = len(self.filters)

        self.upsamples = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=self.upsample_kernel),
                    ConvBNELU_BottleneckGRAM(
                        in_channels=self.in_channels if k == 0 else self.filters[k - 1],
                        out_channels=self.filters[k],
                        kernel_size=self.upsample_kernel,
                        ceil_pad=True,
                        degree=degree,
                        reduction_factor=reduction_factor,
                        dropout_rate=dropout_rate
                    ),
                )
                for k in range(self.depth)
            ]
        )

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBNELU_BottleneckGRAM(
                        in_channels=self.filters[k] * 2,
                        out_channels=self.filters[k],
                        kernel_size=self.kernel_size,
                        ceil_pad=True,
                        degree=degree,
                        reduction_factor=reduction_factor,
                        dropout_rate=dropout_rate
                    )
                )
                for k in range(self.depth)
            ]
        )

    def CropToMatch(self, input: torch.Tensor, shortcut: torch.Tensor) -> torch.Tensor:
        """
        Crop input tensor to match the dimensions of the shortcut tensor.
        
        Args:
            input: Input tensor to be cropped
            shortcut: Reference tensor for dimensions
            
        Returns:
            Cropped tensor matching shortcut dimensions
        """
        diff = max(0, input.shape[2] - shortcut.shape[2])
        start = diff // 2 + diff % 2

        return input[:, :, start : start + shortcut.shape[2]]

    def forward(self, z: torch.Tensor, shortcuts: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the decoder.
        
        Args:
            z: Input tensor from encoder bottleneck
            shortcuts: Skip connections from encoder
            
        Returns:
            Decoded tensor
        """
        for upsample, block, shortcut in zip(
            self.upsamples, self.blocks, shortcuts[::-1]
        ):  # [::-1] data is taken in reverse order
            z = upsample(z)

            if z.shape[2] != shortcut.shape[2]:
                z = self.CropToMatch(z, shortcut)

            z = torch.cat([shortcut, z], dim=1)

            z = block(z)

        return z


class USleep_BottleneckGRAM(nn.Module):
    """
    USleep architecture with Bottleneck GRAM convolutions.
    This implementation significantly reduces parameters while maintaining performance.
    """
    def __init__(
        self,
        num_channels = 2,
        initial_filters = 5,
        complexity_factor = 1.67,
        progression_factor = 2,
        depth = 12,
        num_classes = 5,
        gram_degree = 1,
        reduction_factor = 6,
        dropout_rate = 0.2
    ):
        """
        Initialize USleep architecture with Bottleneck GRAM convolutions.
        
        Args:
            num_channels: Number of input channels
            initial_filters: Initial number of filters
            complexity_factor: Factor for filter growth in width
            progression_factor: Factor for filter growth in depth
            depth: Number of encoder/decoder blocks
            num_classes: Number of output classes
            gram_degree: Maximum degree of Gram polynomials
            reduction_factor: Factor for bottleneck reduction
            dropout_rate: Dropout probability for regularization
        """
        super().__init__()

        self.initial_filters = initial_filters
        self.new_filter_factor = math.sqrt(complexity_factor)
        self.progression_factor = math.sqrt(progression_factor)
        self.depth = depth  # Store depth as instance variable
        
        encoder_filters, decoder_filters, max_filters = self.create_filters()
        
        self.encoder = Encoder_BottleneckGRAM(
            filters=encoder_filters, 
            max_filters=max_filters, 
            in_channels=num_channels,
            degree=gram_degree,
            reduction_factor=reduction_factor,
            dropout_rate=dropout_rate
        )
        
        self.decoder = Decoder_BottleneckGRAM(
            filters=decoder_filters, 
            max_filters=max_filters,
            degree=gram_degree,
            reduction_factor=reduction_factor,
            dropout_rate=dropout_rate
        )
        
        self.dense = Dense_BottleneckGRAM(
            in_channels=encoder_filters[0], 
            num_classes=num_classes,
            reduction_factor=reduction_factor
        )
        
        self.classifier = SegmentClassifier_BottleneckGRAM(
            num_classes=num_classes
        )

    def create_filters(self) -> Tuple[List[int], List[int], int]:
        """
        Create filter configurations for encoder and decoder.
        
        Returns:
            Tuple of (encoder filters, decoder filters, max filters)
        """
        encoder_filters = []
        current_filters = self.initial_filters

        for _ in range(self.depth + 1):  # Use self.depth instead of hardcoded value
            encoder_filters.append(int(current_filters * self.new_filter_factor))
            current_filters = int(self.progression_factor * current_filters)

        max_filters = encoder_filters[-1]
        encoder_filters.pop()
        decoder_filters = encoder_filters[::-1]
        
        return encoder_filters, decoder_filters, max_filters
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the USleep model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, sequence_length)
            
        Returns:
            Classification output tensor
        """
        x, shortcuts = self.encoder(x)
        x = self.decoder(x, shortcuts)
        x = self.dense(x)
        x = self.classifier(x)
        return x


class Dense_BottleneckGRAM(nn.Module):
    """
    Dense layer with optional bottleneck for efficient feature extraction.
    """
    def __init__(
        self, 
        in_channels: int, 
        num_classes: int = 6, 
        kernel_size: int = 1, 
        reduction_factor: int = 4
    ):
        """
        Initialize Dense layer with bottleneck.
        
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            kernel_size: Kernel size for convolution
            reduction_factor: Factor for bottleneck reduction (0 = no bottleneck)
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        
        # Use bottleneck if reduction_factor > 0
        if reduction_factor > 0:
            reduced_channels = max(1, in_channels // reduction_factor)
            self.dense = nn.Sequential(
                # Bottleneck reduction
                nn.Conv1d(
                    in_channels=self.in_channels,
                    out_channels=reduced_channels,
                    kernel_size=1,
                    bias=True,
                ),
                nn.BatchNorm1d(reduced_channels),
                nn.SiLU(),
                # Classification convolution
                nn.Conv1d(
                    in_channels=reduced_channels,
                    out_channels=self.num_classes,
                    kernel_size=self.kernel_size,
                    bias=True,
                ),
                nn.Tanh(),
            )
            # Initialize weights
            nn.init.xavier_uniform_(self.dense[0].weight)
            nn.init.zeros_(self.dense[0].bias)
            nn.init.xavier_uniform_(self.dense[3].weight)
            nn.init.zeros_(self.dense[3].bias)
        else:
            # Standard implementation without bottleneck
            self.dense = nn.Sequential(
                nn.Conv1d(
                    in_channels=self.in_channels,
                    out_channels=self.num_classes,
                    kernel_size=self.kernel_size,
                    bias=True,
                ),
                nn.Tanh(),
            )
            # Initialize weights
            nn.init.xavier_uniform_(self.dense[0].weight)
            nn.init.zeros_(self.dense[0].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Dense layer.
        
        Args:
            z: Input tensor
            
        Returns:
            Output tensor after dense transformation
        """
        return self.dense(z)


class SegmentClassifier_BottleneckGRAM(nn.Module):
    """
    Segment classifier with bottleneck option for the USleep architecture.
    """
    def __init__(
        self, 
        num_classes: int = 5, 
        avgpool_kernel: int = 3840, 
        conv1d_kernel: int = 1,
        reduction_factor: int = 0  # 0 means no bottleneck
    ):
        """
        Initialize Segment Classifier with bottleneck option.
        
        Args:
            num_classes: Number of output classes
            avgpool_kernel: Kernel size for average pooling
            conv1d_kernel: Kernel size for convolution
            reduction_factor: Factor for bottleneck reduction (0 = no bottleneck)
        """
        super().__init__()
        self.num_classes = num_classes
        self.avgpool_kernel = avgpool_kernel
        self.conv1d_kernel = conv1d_kernel

        self.avgpool = nn.AvgPool1d(self.avgpool_kernel)
        
        # Use bottleneck if reduction_factor > 0
        if reduction_factor > 0:
            reduced_channels = max(1, (num_classes + 1) // reduction_factor)
            self.layers = nn.Sequential(
                # First conv with bottleneck
                nn.Conv1d(
                    in_channels=self.num_classes,
                    out_channels=reduced_channels,
                    kernel_size=1,
                ),
                nn.BatchNorm1d(reduced_channels),
                nn.SiLU(),
                nn.Conv1d(
                    in_channels=reduced_channels,
                    out_channels=num_classes,
                    kernel_size=self.conv1d_kernel,
                ),
                # Second conv with bottleneck
                nn.BatchNorm1d(num_classes),
                nn.SiLU(),
                nn.Conv1d(
                    in_channels=num_classes,
                    out_channels=num_classes,
                    kernel_size=self.conv1d_kernel,
                ),
            )
            # Initialize weights
            nn.init.xavier_uniform_(self.layers[0].weight)
            nn.init.zeros_(self.layers[0].bias)
            nn.init.xavier_uniform_(self.layers[3].weight)
            nn.init.zeros_(self.layers[3].bias)
            nn.init.xavier_uniform_(self.layers[6].weight)
            nn.init.zeros_(self.layers[6].bias)
        else:
            # Standard implementation without bottleneck
            self.layers = nn.Sequential(
                nn.Conv1d(
                    in_channels=self.num_classes,
                    out_channels=self.num_classes,
                    kernel_size=self.conv1d_kernel,
                ),
                nn.Conv1d(
                    in_channels=self.num_classes,
                    out_channels=self.num_classes,
                    kernel_size=self.conv1d_kernel,
                ),
            )
            # Initialize weights
            nn.init.xavier_uniform_(self.layers[0].weight)
            nn.init.zeros_(self.layers[0].bias)
            nn.init.xavier_uniform_(self.layers[1].weight)
            nn.init.zeros_(self.layers[1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Segment Classifier.
        
        Args:
            z: Input tensor
            
        Returns:
            Classified segments
        """
        z = self.avgpool(z)
        z = self.layers(z)
        return z


def count_parameters(model: nn.Module) -> int:
    """
    Counts the number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to analyze.

    Returns:
        int: Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_bottleneck_model():
    """
    Test the bottleneck model by creating an instance, counting parameters,
    and measuring forward pass time.
    """
    # Create bottleneck model with recommended parameters
    bottleneck_model = USleep_BottleneckGRAM(
        num_channels=2,
        initial_filters=5,
        complexity_factor=0.25,
        progression_factor=1.5,
        depth=9,
        num_classes=5,
        gram_degree=1,
        reduction_factor=8,
        dropout_rate=0.05
    )
    
    # Count parameters
    bottleneck_params = count_parameters(bottleneck_model)
    
    # Print results
    print(f"Bottleneck USleep_GRAM model: {bottleneck_params:,} parameters")
    
    # Generate sample data for timing comparison
    batch_size = 1
    num_channels = 2
    sequence_length = 4096
    sample_data = torch.randn(batch_size, num_channels, sequence_length)
    
    # Measure forward pass time
    import time
    
    # Bottleneck model timing
    bottleneck_model.eval()
    with torch.no_grad():
        start_time = time.time()
        bottleneck_output = bottleneck_model(sample_data)
        bottleneck_time = time.time() - start_time
    
    print(f"Bottleneck forward pass time: {bottleneck_time:.4f} seconds")
    
    # Output shape
    print(f"Bottleneck output shape: {bottleneck_output.shape}")


if __name__ == "__main__":
    # Test the bottleneck model
    test_bottleneck_model()
