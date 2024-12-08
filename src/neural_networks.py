# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 13:01:11 2024

This script contains the neural networks that will be used for the "neural
intuition" part of the framework.

It would be interesting to have first some sort of projection from the grid
spaces into an embedding (for each input-output pair), and then an Attention
module going aggregating the embeddings to go to the actual multi-label
classification part. The reasoning behind this is: sometimes it's impossible to
guess what the rule behind the task is just from one example, multiple examples
at the same time should be considered.

@author: Alberto
"""
import torch

class EmbedInputOutputPair(torch.nn.Module) :
    """
    Since the problem has to do with spatial representation, we will need some
    set of convolutional parts, with the classic convolution+activation+downscaling part;
    Note that there is no normalization at the end, as we are going to use a
    BCELoss that requires the unnormalized outputs (logits) of the network.
    """
    def __init__(self, n_classes=190) :
        super(EmbedInputOutputPair, self).__init__()
        self.features = torch.nn.Sequential(
            # these are two copies of a classic sequence of modules copied by VGG-16
            torch.nn.Conv2d(1, 32, kernel_size=(2,2), stride=(1,1), padding=(1,1)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 32, kernel_size=(2,2), stride=(1,1), padding=(1,1)),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            
            torch.nn.Conv2d(32, 64, kernel_size=(2,2), stride=(1,1), padding=(1,1)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 128, kernel_size=(2,2), stride=(1,1), padding=(1,1)),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            )
        self.fully_connected_1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=128, out_features=256, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(in_features=256, out_features=n_classes, bias=True)
            )
    
    def forward(self, x) :
        z_1 = self.features(x)
        z_2 = self.fully_connected_1(z_1)
        
        return z_2
    

class SiameseMultiLabelNetwork(torch.nn.Module):
    def __init__(self, n_classes=190, input_channels=1):
        """
        Siamese network for multi-label classification of paired images
        
        Args:
            num_classes (int): Number of labels to predict
            input_channels (int): Number of input image channels (default: 1)
        """
        super(SiameseMultiLabelNetwork, self).__init__()
        
        # Shared Convolutional Base
        self.conv_base = torch.nn.Sequential(
            # First convolutional layer
            torch.nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            
            # Second convolutional layer
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            
            # Third convolutional layer
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )
        
        # Calculate the flattened size after convolutions
        # For 30x30 input, this will be approximately 128 * 3 * 3 = 1152
        self.fc_size = 128 * 3 * 3
        
        # Fusion and Classification Layers
        self.fusion_layers = torch.nn.Sequential(
            torch.nn.Linear(self.fc_size * 2, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5)
        )
        
        # Multi-label classification head
        self.classification_head = torch.nn.Linear(256, n_classes)
    
    def forward_once(self, x):
        """Process a single input through the convolutional base"""
        x = self.conv_base(x)
        x = x.view(-1, self.fc_size)
        return x
    
    def forward(self, input1, input2):
        """
        Forward pass for two input images
        
        Args:
            input1 (tensor): First image tensor (B x C x H x W)
            input2 (tensor): Second image tensor (B x C x H x W)
        
        Returns:
            tensor: Multi-label classification probabilities
        """
        # Extract features from both inputs
        feat1 = self.forward_once(input1)
        feat2 = self.forward_once(input2)
        
        # Concatenate features
        combined_features = torch.cat((feat1, feat2), dim=1)
        
        # Pass through fusion layers
        fused_features = self.fusion_layers(combined_features)
        
        # Multi-label classification (sigmoid for multi-label)
        output = torch.sigmoid(self.classification_head(fused_features))
        
        return output
    
class GridPairsDataset(torch.utils.data.Dataset) :
    """
    This class implements a pytorch Dataset which will return two separate
    input/output grid pairs, associated to a multi-label output.
    """
    def __init__(self, input_grids, output_grids, y) :
        # 'data' is a series of tuples (input_grid, output_grid, class_labels)
        self.input_grids = input_grids
        self.output_grids = output_grids
        self.y = y
    
    def __len__(self) :
        return self.y.shape[0]
    
    def __getitem__(self, idx) :
        input_grid = self.input_grids[idx]
        output_grid = self.output_grids[idx]
        y = self.y[idx]
        
        return (
            torch.tensor(input_grid, dtype=torch.float32),
            torch.tensor(output_grid, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
            )
    
    
if __name__ == "__main__" :
    
    # this is just to check that everything is in order
    neural_network = EmbedInputOutputPair(n_classes=8)
    print(neural_network)
    
    # another test
    num_classes = 5  
    siamese_network = SiameseMultiLabelNetwork(num_classes, input_channels=1)

    # Print model summary
    print(siamese_network)
