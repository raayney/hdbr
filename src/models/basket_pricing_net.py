import torch
import torch.nn as nn

class BasketPricingNet(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dims=None, dropout: float = 0.2):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        layers = []
        prev_dim = input_dim
        
        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if i < 2 and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        
        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 1) 
    
    def forward(self, x):
        h = self.feature_extractor(x)
        out = self.output_layer(h)
        return out.squeeze(-1)
