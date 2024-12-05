from typing import Union
import logging
import torch
import torch.nn as nn
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)

class BaseMLP(nn.Module):
    def __init__(self, input_dim, num_layers=2, cond_dim=None, diffusion_step_embed_dim=256, hidden_dim=512):
        super().__init__()

        # Time encoding
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        # Calculate total input dimension (including condition) for the first layer
        self.input_dim = input_dim + diffusion_step_embed_dim + 38  # Assuming global_cond has dim 38
        
        # Define layers
        self.layers = nn.ModuleList([
            nn.Linear(self.input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.activation = nn.Mish()
        self.final_fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, timestep, global_cond=None):
        timesteps = timestep
        
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=x.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(x.device)

        timesteps = timesteps.expand(x.shape[0])

        # Encode diffusion step
        global_feature = self.diffusion_step_encoder(timesteps)

        # Concatenate global condition with input once
        if global_cond is not None:
            condition = torch.cat([global_feature, global_cond], dim=-1)
            condition_expanded = condition.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = torch.cat([x, condition_expanded], dim=-1)

        # Pass through layers
        for fc_layer in self.layers:
            x = self.activation(fc_layer(x))

        return self.final_fc(x)
