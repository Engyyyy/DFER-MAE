import torch
import torch.nn as nn

from models_mae_encoder import mae_encoder_base_patch16, mae_encoder_large_patch16, mae_encoder_huge_patch14

class TunedMae(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        
        self.encoder = encoder
        self.fc = nn.Linear(197 * 1024, 7)
        
    def forward(self, x):
        encoded = self.encoder(x)
        flattened = torch.flatten(encoded, start_dim=1)
        out = self.fc(flattened)
        return out
    

def tuned_mae_base_patch16_dec512d8b(**kwargs):
    model = TunedMae(mae_encoder_base_patch16(), **kwargs)
    return model

def tuned_mae_large_patch16_dec512d8b(**kwargs):
    model = TunedMae(mae_encoder_large_patch16(), **kwargs)
    return model

def tuned_mae_huge_patch14_dec512d8b(**kwargs):
    model = TunedMae(mae_encoder_huge_patch14(), **kwargs)
    return model

def tuned_mae_custom_dec512d8b(**kwargs):
    model = TunedMae(**kwargs)
    return model


tuned_mae_base_patch16 = tuned_mae_base_patch16_dec512d8b
tuned_mae_large_patch16 = tuned_mae_large_patch16_dec512d8b
tuned_mae_huge_patch14 = tuned_mae_huge_patch14_dec512d8b
tuned_mae_custom = tuned_mae_custom_dec512d8b