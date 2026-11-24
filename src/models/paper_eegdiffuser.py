import math
from typing import Callable, Optional
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1) #old
    # Paper Eq. (7): gamma * LN(h) + beta
    return scale.unsqueeze(1) * x + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # unchanged
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        # unchanged
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels: torch.Tensor, force_drop_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Drops labels to enable classifier-free guidance.
        """
        # unchanged
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        # unchanged
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class PatchEmbed(nn.Module):

    def __init__(
            
            self,
            #time_points: Optional[int] = 800, #unused
            patch_size: int = 5,
            in_chans: int = 22,
            embed_dim: int = 512, # code says  256
            norm_layer: Optional[Callable] = None,
            bias: bool = True,
    ):
        super().__init__()
        # unchanged
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size,
                              bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # function unchanged
        # x: (B, C, L) -> (B, L // patch_size, D)
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x



class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0,  **block_kwargs):
        super().__init__()
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True, norm_layer=nn.LayerNorm, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        # Paper Eq. (6): accepts concatenated [e_y; e_t] of dimension 2 * hidden_size
        # since we are concatinating two list of the same size it is doubled 
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(2 * hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Unchanged
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        # print(x.shape, gate_msa.shape, shift_msa.shape, scale_msa.shape)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x



class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            # Concatinated dimensions
            nn.Linear(2 * hidden_size, 2 * hidden_size, bias=True),
        )
        # Paper Eq. (9) change: ConvTranspose1D maps patch tokens back to (B, out_channels, L)
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels=hidden_size,
            out_channels=out_channels,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=True,
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        # Paper Eq. (9) change
        x = self.conv_transpose(x.transpose(1, 2))
        return x


class EEGDiffuser(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        in_channels: int,
        time_points: int,
        num_classes: int,
        embed_dim: int = 512,
        patch_size: int = 5,
        depth: int = 4,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        learn_sigma: bool = False, #for VESDE it should be false
    ):
        super().__init__()
        self.in_channels = in_channels
        self.time_points = time_points
        self.patch_size = patch_size
        self.learn_sigma = learn_sigma
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.sequence_len = time_points // patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(
            patch_size=patch_size, 
            in_chans=in_channels, 
            embed_dim=embed_dim)
        self.t_embedder = TimestepEmbedder(embed_dim)
        self.y_embedder = LabelEmbedder(num_classes, embed_dim, class_dropout_prob)
        self.pos_embed = nn.Parameter(torch.empty(1, self.sequence_len, embed_dim).normal_(std=0.02))

        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(embed_dim, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(22 ** 0.5))
        # print(pos_embed.shape, self.pos_embed.data.shape)
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))


        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)


        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        # Paper Eq. (9) change
        nn.init.constant_(self.final_layer.conv_transpose.weight, 0)
        nn.init.constant_(self.final_layer.conv_transpose.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        B, C, P = x.shape
        # print(x.shape)
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        # print(x.shape)
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        y = self.y_embedder(y, self.training)   # (N, D)
        # Paper Eq. (6): concatenation [e_y; e_t]
        c = torch.cat([y, t], dim=-1)

        for block in self.blocks:
            x = block(x, c)
        return self.final_layer(x, c)

    def forward_with_cfg(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, cfg_scale: float) -> torch.Tensor:
        # Paper Eq. (4): classifier-free guidance across full output score/noise channels
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)

        if self.learn_sigma:
            eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            guided_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([guided_eps, guided_eps], dim=0)
            return torch.cat([eps, rest], dim=1)

        cond_out, uncond_out = torch.split(model_out, len(model_out) // 2, dim=0)
        half_eps = uncond_out + cfg_scale * (cond_out - uncond_out)
        return torch.cat([half_eps, half_eps], dim=0)