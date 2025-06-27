
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LeNet5(nn.Module):
    
    def __init__(self, num_classes=33, input_channels=1):
        """
        Initialize LeNet-5 model
        
        Args:
            num_classes (int): Number of output classes (33 for Tifinagh)
            input_channels (int): Number of input channels (1 for grayscale)
        """
        super(LeNet5, self).__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        # C1: First Convolutional Layer
        # Input: 1×32×32 → Output: 6×28×28
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=6,
            kernel_size=5,
            stride=1,
            padding=0,
            bias=True
        )
        
        # S2: First Subsampling Layer (Max Pooling)
        # Input: 6×28×28 → Output: 6×14×14
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # C3: Second Convolutional Layer
        # Input: 6×14×14 → Output: 16×10×10
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0,
            bias=True
        )
        
        # S4: Second Subsampling Layer (Max Pooling)
        # Input: 16×10×10 → Output: 16×5×5
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # FC1: First Fully Connected Layer
        # Input: 16×5×5 = 400 → Output: 120
        self.fc1 = nn.Linear(16 * 5 * 5, 120, bias=True)
        
        # FC2: Second Fully Connected Layer
        # Input: 120 → Output: 84
        self.fc2 = nn.Linear(120, 84, bias=True)
        
        # Output Layer
        # Input: 84 → Output: num_classes (33)
        self.fc3 = nn.Linear(84, num_classes, bias=True)
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Validate input dimensions
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor, got {x.dim()}D")
        
        if x.size(1) != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, got {x.size(1)}")
        
        if x.size(2) != 32 or x.size(3) != 32:
            raise ValueError(f"Expected 32×32 input size, got {x.size(2)}×{x.size(3)}")
        
        # C1: First Convolutional Layer + ReLU
        x = F.relu(self.conv1(x))  # Shape: (batch_size, 6, 28, 28)
        
        # S2: First Pooling Layer
        x = self.pool1(x)  # Shape: (batch_size, 6, 14, 14)
        
        # C3: Second Convolutional Layer + ReLU
        x = F.relu(self.conv2(x))  # Shape: (batch_size, 16, 10, 10)
        
        # S4: Second Pooling Layer
        x = self.pool2(x)  # Shape: (batch_size, 16, 5, 5)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Shape: (batch_size, 400)
        
        # FC1: First Fully Connected Layer + ReLU
        x = F.relu(self.fc1(x))  # Shape: (batch_size, 120)
        
        # FC2: Second Fully Connected Layer + ReLU
        x = F.relu(self.fc2(x))  # Shape: (batch_size, 84)
        
        # Output Layer (no activation, raw logits)
        x = self.fc3(x)  # Shape: (batch_size, num_classes)
        
        return x
    
    def _initialize_weights(self):
        """
        Initialize network weights using Xavier/Glorot initialization
        
        Mathematical Formula:
        For layer with n_in inputs and n_out outputs:
        W ~ Uniform(-√(6/(n_in + n_out)), √(6/(n_in + n_out)))
        b = 0
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Xavier initialization for convolutional layers
                n_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                n_out = module.out_channels * module.kernel_size[0] * module.kernel_size[1]
                limit = math.sqrt(6 / (n_in + n_out))
                nn.init.uniform_(module.weight, -limit, limit)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.Linear):
                # Xavier initialization for fully connected layers
                n_in, n_out = module.weight.size(1), module.weight.size(0)
                limit = math.sqrt(6 / (n_in + n_out))
                nn.init.uniform_(module.weight, -limit, limit)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def get_feature_maps(self, x, layer_name='conv1'):
        """
        Extract feature maps from intermediate layers for visualization
        
        Args:
            x (torch.Tensor): Input tensor
            layer_name (str): Layer name ('conv1', 'conv2', 'pool1', 'pool2')
            
        Returns:
            torch.Tensor: Feature maps from specified layer
        """
        if layer_name == 'conv1':
            return F.relu(self.conv1(x))
        elif layer_name == 'pool1':
            x = F.relu(self.conv1(x))
            return self.pool1(x)
        elif layer_name == 'conv2':
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            return F.relu(self.conv2(x))
        elif layer_name == 'pool2':
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            return self.pool2(x)
        else:
            raise ValueError(f"Unknown layer name: {layer_name}")
    
    def get_conv_filters(self, layer_name='conv1'):
        """
        Get convolutional filter weights for visualization
        
        Args:
            layer_name (str): Layer name ('conv1' or 'conv2')
            
        Returns:
            torch.Tensor: Filter weights
        """
        if layer_name == 'conv1':
            return self.conv1.weight.data  # Shape: (6, 1, 5, 5)
        elif layer_name == 'conv2':
            return self.conv2.weight.data  # Shape: (16, 6, 5, 5)
        else:
            raise ValueError(f"Unknown layer name: {layer_name}")
    
    def count_parameters(self):
        """
        Count the total number of trainable parameters
        
        Returns:
            int: Total number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_layer_info(self):
        """
        Get detailed information about each layer
        
        Returns:
            dict: Layer information including parameters and output shapes
        """
        info = {
            'conv1': {
                'type': 'Conv2d',
                'input_shape': (1, 32, 32),
                'output_shape': (6, 28, 28),
                'parameters': 6 * (5 * 5 * 1 + 1),
                'kernel_size': (5, 5),
                'filters': 6
            },
            'pool1': {
                'type': 'MaxPool2d',
                'input_shape': (6, 28, 28),
                'output_shape': (6, 14, 14),
                'parameters': 0,
                'kernel_size': (2, 2)
            },
            'conv2': {
                'type': 'Conv2d',
                'input_shape': (6, 14, 14),
                'output_shape': (16, 10, 10),
                'parameters': 16 * (5 * 5 * 6 + 1),
                'kernel_size': (5, 5),
                'filters': 16
            },
            'pool2': {
                'type': 'MaxPool2d',
                'input_shape': (16, 10, 10),
                'output_shape': (16, 5, 5),
                'parameters': 0,
                'kernel_size': (2, 2)
            },
            'fc1': {
                'type': 'Linear',
                'input_size': 400,
                'output_size': 120,
                'parameters': 400 * 120 + 120
            },
            'fc2': {
                'type': 'Linear',
                'input_size': 120,
                'output_size': 84,
                'parameters': 120 * 84 + 84
            },
            'fc3': {
                'type': 'Linear',
                'input_size': 84,
                'output_size': self.num_classes,
                'parameters': 84 * self.num_classes + self.num_classes
            }
        }
        
        total_params = sum(layer['parameters'] for layer in info.values())
        info['total_parameters'] = total_params
        
        return info


def test_model():
    """
    Test the LeNet-5 model implementation
    """
    print("Testing LeNet-5 Model Implementation")
    print("=" * 50)
    
    # Create model
    model = LeNet5(num_classes=33)
    
    # Print model information
    layer_info = model.get_layer_info()
    total_params = model.count_parameters()
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Expected Parameters: 63,661")
    print(f"Match: {total_params == 63661}")
    print()
    
    # Test forward pass
    batch_size = 4
    test_input = torch.randn(batch_size, 1, 32, 32)
    
    print(f"Input shape: {test_input.shape}")
    
    # Forward pass
    output = model(test_input)
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, 33)")
    print(f"Match: {output.shape == (batch_size, 33)}")
    print()
    
    # Test feature map extraction
    feature_maps_conv1 = model.get_feature_maps(test_input, 'conv1')
    print(f"Conv1 feature maps shape: {feature_maps_conv1.shape}")
    print(f"Expected: ({batch_size}, 6, 28, 28)")
    
    # Test filter visualization
    conv1_filters = model.get_conv_filters('conv1')
    print(f"Conv1 filters shape: {conv1_filters.shape}")
    print(f"Expected: (6, 1, 5, 5)")
    
    print("\nModel test completed successfully!")


if __name__ == "__main__":
    test_model()