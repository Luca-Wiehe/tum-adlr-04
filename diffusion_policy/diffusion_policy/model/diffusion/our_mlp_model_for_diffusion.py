from typing import Union
import logging
import torch
import torch.nn as nn

from diffusion_policy.model.diffusion.our_model_mlp_components import (
   DownsampleMLP, UpsampleMLP, MLPBlock
)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)

# class MLPDiffusion(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.stack = nn.Sequential(
#             nn.Linear(input_dim, 512),
#             nn.Mish(),
#             nn.Linear(512, 512),
#             nn.Mish(),
#             nn.Linear(512, output_dim)
#         )

#     def forward(self, o):
#         o = self.flatten(o)
#         logits = self.stack(o)
#         return logits

class ConditionalResidualBlockMLP(nn.Module):
    def __init__(self, inp_dim, out_dim, cond_dim, n_groups=8, cond_predict_scale=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            MLPBlock(inp_dim, out_dim, n_groups=n_groups),
            MLPBlock(out_dim, out_dim, n_groups=n_groups),
        ])

        # FiLM modulation
        cond_channels = out_dim * 2 if cond_predict_scale else out_dim
        self.cond_predict_scale = cond_predict_scale
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
        )

        # Ensure residual dimensions match
        self.residual_fc = nn.Linear(inp_dim, out_dim) if inp_dim != out_dim else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            scale, bias = embed.chunk(2, dim=-1)
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        return out + self.residual_fc(x)


class ConditionalUnetMLP(nn.Module):
    def __init__(self, input_dim, local_cond_dim=None, global_cond_dim=None, diffusion_step_embed_dim=256,
                 down_dims=[256, 512, 1024], n_groups=8, cond_predict_scale=False):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        # Time encoding
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        cond_dim = diffusion_step_embed_dim
        if global_cond_dim:
            cond_dim += global_cond_dim

        # Down and up sampling
        self.down_modules = nn.ModuleList([
            nn.ModuleList([
                ConditionalResidualBlockMLP(dim_in, dim_out, cond_dim, n_groups, cond_predict_scale),
                DownsampleMLP(dim_out),
            ]) for dim_in, dim_out in zip(all_dims[:-1], all_dims[1:])
        ])

        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlockMLP(down_dims[-1], down_dims[-1], cond_dim, n_groups, cond_predict_scale),
            ConditionalResidualBlockMLP(down_dims[-1], down_dims[-1], cond_dim, n_groups, cond_predict_scale),
        ])

        self.up_modules = nn.ModuleList([
            nn.ModuleList([
                ConditionalResidualBlockMLP(dim_out, dim_in, cond_dim, n_groups, cond_predict_scale),
                UpsampleMLP(dim_in),
            ]) for dim_in, dim_out in zip(reversed(all_dims[:-1]), reversed(all_dims[1:]))
        ])

        self.final_fc = nn.Sequential(
            MLPBlock(start_dim, start_dim, n_groups),
            nn.Linear(start_dim, input_dim),
        )

    def forward(self, x, timestep, global_cond=None):
        timesteps = self.diffusion_step_encoder(timestep)

        if global_cond is not None:
            timesteps = torch.cat([timesteps, global_cond], dim=-1)

        # Downward path
        h = []
        for res_block, downsample in self.down_modules:
            x = res_block(x, timesteps)
            h.append(x)
            x = downsample(x)

        # Middle layers
        for mid_block in self.mid_modules:
            x = mid_block(x, timesteps)

        # Upward path
        for res_block, upsample in self.up_modules:
            x = torch.cat([x, h.pop()], dim=-1)
            x = res_block(x, timesteps)
            x = upsample(x)

        return self.final_fc(x)
