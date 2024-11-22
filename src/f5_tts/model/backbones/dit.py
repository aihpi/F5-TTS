"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from x_transformers.x_transformers import RotaryEmbedding

from f5_tts.model.modules import (
    TimestepEmbedding,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    AdaLayerNormZero_Final,
    precompute_freqs_cis,
    get_pos_embed_indices,
)


# Text embedding
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class TokenCharFusion(nn.Module):
    def __init__(self, token_dim, text_dim, eps=1e-6):
        super().__init__()
        # RMSNorm
        self.norm_pre = RMSNorm(token_dim, eps=eps)
        self.norm_token = RMSNorm(text_dim, eps=eps)
        self.char_norm = RMSNorm(text_dim, eps=eps)

        # Down-projection and processing
        self.token_down = nn.Linear(token_dim, text_dim, bias=False)
        self.activation = nn.SiLU()
        self.token_proj = nn.Linear(text_dim, 3, bias=False)  # scale, shift, gate
        self.fusion_linear = nn.Linear(text_dim, text_dim, bias=False)

        # Initialize
        nn.init.xavier_uniform_(self.token_down.weight)
        nn.init.xavier_uniform_(self.token_proj.weight, gain=0.2)
        nn.init.xavier_uniform_(self.fusion_linear.weight)

    def forward(
            self,
            token_embeds: torch.Tensor,  # LLM embeddings, shape: (batch_size, n_token, token_dim)
            token_ids_mask: torch.Tensor,
            char_embeds: torch.Tensor,  # Shape: (batch_size, n_token, n_char, text_dim)
            char_ids_mask: torch.Tensor,
            filler_embed: torch.Tensor
    ):
        # Initial normalization of LLM embeddings
        token_embeds = self.norm_pre(token_embeds)

        # Project and normalize
        token_feats = self.token_down(token_embeds)
        token_feats = self.norm_token(token_feats)

        # Generate modulation params
        token_params = self.token_proj(self.activation(token_feats))
        scale, shift, gate = torch.chunk(token_params, 3, dim=-1)

        # Normalize character embeddings
        char_embeds_norm = self.char_norm(char_embeds)

        # Expand and modulate
        scale = scale.unsqueeze(2)  # Shape: (batch_size, n_token, 1, 1)
        shift = shift.unsqueeze(2)  # Shape: (batch_size, n_token, 1, 1)
        gate = gate.unsqueeze(2)  # Shape: (batch_size, n_token, 1, 1)

        # AdaLN-style modulation
        fused_embeds = char_embeds_norm * (1 + scale) + shift

        # Gated projection
        fused_embeds = self.activation(fused_embeds)
        fused_embeds = self.fusion_linear(fused_embeds)

        # Compute gate
        gate = gate.sigmoid()

        # Blend fused embeddings and original character embeddings
        fused_embeds = gate * fused_embeds + (1 - gate) * char_embeds_norm

        # Flatten fused embeddings and char_ids_mask
        batch_size, n_token, n_char, d_fusion = fused_embeds.shape
        fused_embeds_flat = fused_embeds.view(batch_size, n_token * n_char, d_fusion)  # b x (n_token * n_char) x d_fusion
        char_ids_mask_flat = char_ids_mask.view(batch_size, n_token * n_char)  # b x (n_token * n_char)

        # Mask valid characters and reshape
        valid_char_embeds = fused_embeds_flat[char_ids_mask_flat.bool()]  # [total_valid_chars, d_fusion]

        # Create padded output for consistent batch size
        max_valid_chars = char_ids_mask_flat.sum(dim=1).max().item()  # Maximum valid characters in any batch
        padded_char_embeds = filler_embed.expand(batch_size, max_valid_chars, d_fusion).clone()  # Use filler embedding

        # Gather valid character embeddings back into padded structure
        valid_char_indices = char_ids_mask_flat.cumsum(dim=1) - 1  # Offset indices for valid chars
        batch_offsets = torch.arange(batch_size, device=fused_embeds.device).unsqueeze(1) * max_valid_chars
        valid_char_indices = batch_offsets + valid_char_indices[char_ids_mask_flat.bool()]
        padded_char_embeds.view(-1, d_fusion)[valid_char_indices] = valid_char_embeds

        return padded_char_embeds  # b x max_valid_chars x d_fusion


class TextEmbedding(nn.Module):
    def __init__(
            self,
            text_num_embeds,
            text_dim,
            token_num_embeds,
            token_dim,
            pad_token_id,
            conv_layers=0,
            conv_mult=2
    ):
        super().__init__()
        # text_embed is for characters
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token
        self.token_embed = nn.Embedding(token_num_embeds, token_dim)
        self.pad_token_id = pad_token_id
        self.fusion = TokenCharFusion(token_dim, text_dim)

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(
            self,
            token_ids: torch.LongTensor,  # b x n_token
            token_ids_mask: torch.LongTensor,  # b x n_token
            char_ids: torch.LongTensor,  # b x n_token x n_char
            char_ids_mask: torch.LongTensor,  # b x n_token x n_char
            seq_len,
            drop_text=False
    ):  # noqa: F722
        char_ids = char_ids + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx() / collate_fn
        # Use a dummy tensor as filler input for the embeddings
        filler_char_id = torch.zeros(1, 1, dtype=torch.long, device=char_ids.device)
        filler_embed = self.text_embed(filler_char_id).squeeze(0)  # 1 x d_char

        if drop_text:  # cfg for text
            batch_size, n_token, n_char = char_ids_mask.shape
            char_ids_mask_flat = char_ids_mask.view(batch_size, n_token * n_char)
            max_valid_chars = char_ids_mask_flat.sum(dim=1).max().item()
            char_ids = torch.zeros(batch_size, max_valid_chars, dtype=torch.long, device=char_ids.device)
            text = self.text_embed(char_ids)
        else:
            char_embeds = self.text_embed(char_ids)  # b n -> b n_token n_char d_char
            with torch.no_grad():
                token_embeds = self.token_embed(token_ids)  # b n  -> b n_token d_token
            text = self.fusion(token_embeds, token_ids_mask, char_embeds, char_ids_mask, filler_embed)

        # Truncate or pad along the seq_len dimension if necessary
        text = text[:, :seq_len, :]
        batch, text_len, embed_dim = text.shape
        if text_len < seq_len:
            padding_len = seq_len - text_len
            filler_padding = filler_embed.unsqueeze(0).expand(batch, padding_len, -1)
            text = torch.cat([text, filler_padding], dim=1)

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnextv2 blocks
            text = self.text_blocks(text)

        return text


# noised input audio and context mixing embedding


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: float["b n d"], cond: float["b n d"], text_embed: float["b n d"], drop_audio_cond=False):  # noqa: F722
        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)

        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


# Transformer backbone using DiT blocks


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        token_num_embeds=None,
        token_dim=None,
        pad_token_id=None,
        conv_layers=0,
        long_skip_connection=False,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(
            text_num_embeds=text_num_embeds,
            text_dim=text_dim,
            token_num_embeds=token_num_embeds,
            token_dim=token_dim,
            pad_token_id=pad_token_id,
            conv_layers=conv_layers
        )
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout) for _ in range(depth)]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

    def set_token_embeddings(self, value):
        self.text_embed.token_embed = value

    def forward(
        self,
        x: float["b n d"],  # nosied input audio  # noqa: F722
        cond: float["b n d"],  # masked cond audio  # noqa: F722
        token_ids: torch.LongTensor, # b x n_token
        token_ids_mask: torch.LongTensor, # b x n_token
        char_ids: torch.LongTensor, # b x n_token x n_char
        char_ids_mask: torch.LongTensor, # b x n_token x n_char
        time: float["b"] | float[""],  # time step  # noqa: F821 F722
        drop_audio_cond,  # cfg for cond audio
        drop_text,  # cfg for text
        mask: bool["b n"] | None = None,  # noqa: F722
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(time)
        text_embed = self.text_embed(token_ids, token_ids_mask, char_ids, char_ids_mask, seq_len, drop_text=drop_text)
        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output
