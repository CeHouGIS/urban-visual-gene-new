"""Frozen, local DINOv3 ViT-L/16 with explicit prefix-token exclusion."""
import torch
from torch import nn


class FrozenDINO(nn.Module):
    def __init__(self, weights, resolution=512):
        super().__init__()
        from transformers import AutoModel
        self.model, info = AutoModel.from_pretrained(
            weights, local_files_only=True, output_loading_info=True, torch_dtype=torch.bfloat16)
        if any(info.get(k) for k in ('missing_keys', 'mismatched_keys', 'unexpected_keys', 'error_msgs')):
            raise RuntimeError(f'Incomplete pretrained model: {info}')
        c = self.model.config
        if (c.model_type, c.hidden_size, c.patch_size) != ('dinov3_vit', 1024, 16):
            raise ValueError('This protocol requires DINOv3 ViT-L/16, 1024 channels')
        self.resolution = resolution
        self.side = resolution // c.patch_size
        self.prefix = 1 + c.num_register_tokens
        self.model.eval().requires_grad_(False)
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, images):
        if not images.is_cuda:
            raise RuntimeError('DINO neural inference requires a GPU')
        if images.shape[1:] != (3, self.resolution, self.resolution):
            raise ValueError('Unexpected RGB input shape')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            tokens = self.model(pixel_values=(images-self.mean)/self.std).last_hidden_state
        if tokens.shape[1:] != (self.prefix + self.side*self.side, 1024):
            raise ValueError(f'Unexpected DINO token shape {tokens.shape}')
        return tokens[:, self.prefix:].transpose(1, 2).reshape(len(images), 1024, self.side, self.side).float()
