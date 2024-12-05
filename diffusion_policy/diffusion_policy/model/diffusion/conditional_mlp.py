from typing import Union
import logging
import torch
import torch.nn as nn

from diffusion_policy.model.diffusion.mlp_components import (
   DownsampleMLP, UpsampleMLP, MLPBlock
)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from diffusion_policy.model.diffusion.mlp_components import MLPBlock
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

class ConditionalFiLMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, cond_dim, cond_predict_scale=False):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.cond_predict_scale = cond_predict_scale

        # Use cond_dim for the conditioning input
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, output_dim * 2 if cond_predict_scale else output_dim),
        )

    def forward(self, x, cond):
        # Apply the main fully connected layer
        out = self.fc(x)  # Shape: [batch_size, seq_len, output_dim]

        # Generate conditional embeddings
        embed = self.cond_encoder(cond)  # Shape: [batch_size, output_dim * 2] or [batch_size, output_dim]

        # Expand conditioning embeddings across the sequence dimension
        embed = embed.unsqueeze(1).expand(-1, x.shape[1], -1)  # Shape: [batch_size, seq_len, output_dim * 2] or [batch_size, seq_len, output_dim]

        # FiLM conditioning
        if self.cond_predict_scale:
            scale, bias = embed.chunk(2, dim=-1)  # Split into scale and bias
            out = scale * out + bias
        else:
            out = out + embed

        return out


class ConditionalMLP(nn.Module):
    def __init__(self, input_dim, num_layers=2, cond_dim=None, diffusion_step_embed_dim=256, hidden_dim=512, cond_predict_scale=False):
        super().__init__()

        # Time encoding
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        self.cond_dim = diffusion_step_embed_dim + 38
        
        self.layers = nn.ModuleList([
            ConditionalFiLMLayer(input_dim if i == 0 else hidden_dim, hidden_dim, self.cond_dim, cond_predict_scale)
            for i in range(num_layers)
        ])
        self.final_fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, timestep, global_cond=None):
        timesteps = timestep
        
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=x.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(x.device)

        timesteps = timesteps.expand(x.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            condition = torch.cat([global_feature, global_cond], dim=-1)

        for fc_layer in self.layers:
            x = fc_layer(x, condition)

        return self.final_fc(x)

