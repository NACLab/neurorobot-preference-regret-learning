"""
File: transformers.py
Author: Viet Nguyen
Date: 2025-02-16

Description:
"""

from typing import Callable
import jax
import jax.numpy as jnp
import numpy as np
import flax

from ..cv.mask_ops import take_patches_by_masks
from .. import nn

# class AttentionMask:

#   def __init__(self, mask: jax.Array):
#     self.mask = mask

#   def __call__(self, x: jax.Array, mask=None, training=True):
#     if mask is None:
#       return None
#     return mask


class Transformer(nn.Module):

  units: int = 256
  layers: int = 8
  heads: int = 8
  ffup: int = 4
  act: str = 'gelu_tanh'
  norm: str = 'layer'
  qknorm: str = 'none'
  dropout: float = 0.0
  bias: bool = True
  winit: str | Callable = nn.Initializer('trunc_normal')
  binit: str | Callable = nn.Initializer('zeros')
  outscale: float = 1.0

  def __call__(self, x: jax.Array, mask=None, training=True):
    lkw = {k: getattr(self, k) for k in ('bias', 'winit', 'binit')}
    akw = {k: getattr(self, k) for k in ('heads', 'qknorm', 'outscale', 'dropout')}
    B, T, D = x.shape
    if D != self.units:
      x = self.sub('proj', nn.Linear, self.units, **lkw)(x)
    for i in range(self.layers):
      with nn.scope(f'layer{i}'):
        skip = x
        x = self.sub('norm1', nn.Norm, self.norm)(x)
        x  = self.sub('mha', nn.Attention, self.units, **lkw, **akw)(x, x, mask, training)
        x += skip
        skip = x
        x = self.sub('norm2', nn.Norm, self.norm)(x)
        ff1 = self.sub('ff1', nn.Linear, self.units * self.ffup, **lkw)
        ff2 = self.sub('ff2', nn.Linear, self.units, **lkw, outscale=self.outscale)
        x = ff2(nn.get_act(self.act)(ff1(x)))
        x += skip
    x = self.sub('outnorm', nn.Norm, self.norm)(x)
    return x


class TransformerWithCondition(nn.Module):

  units: int = 1024
  layers: int = 12
  heads: int = 8
  ffup: int = 4
  act: str = 'gelu_tanh'
  norm: str = 'layer'
  qknorm: str = 'none'
  dropout: float = 0.2
  bias: bool = True
  winit: str | Callable = nn.Initializer('trunc_normal')
  binit: str | Callable = nn.Initializer('zeros')
  outscale: float = 1.0

  def __call__(self, x: jax.Array, cond: jax.Array, mask=None, training=True):
    lkw = {k: getattr(self, k) for k in ('bias', 'winit', 'binit')}
    akw = {k: getattr(self, k) for k in ('heads', 'qknorm', 'outscale', 'dropout')}
    if x.shape[-1] != self.units:
      x = self.sub('projx', nn.Linear, self.units, **lkw)(x)
    if x.shape[-1] != self.units:
      cond = self.sub('projc', nn.Linear, self.units, **lkw)(cond)
    cond = self.sub('normc', nn.Norm, self.norm)(cond)
    for i in range(self.layers):
      with nn.scope(f'layer{i}'):
        skip = x
        x = self.sub('norm1', nn.Norm, self.norm)(x)
        x  = self.sub('mha', nn.Attention, self.units, **lkw, **akw)(x, cond, mask, training)
        x += skip
        skip = x
        x = self.sub('norm2', nn.Norm, self.norm)(x)
        ff1 = self.sub('ff1', nn.Linear, self.units * self.ffup, **lkw)
        ff2 = self.sub('ff2', nn.Linear, self.units, **lkw, outscale=self.outscale)
        x = ff2(nn.get_act(self.act)(ff1(x)))
        x += skip
    x = self.sub('outnorm', nn.Norm, self.norm)(x)
    return x


class ResidualBlock(nn.Module):

  """based on ViT repo
    For each residual stage, perform stride (linear downscaling) in the first layer (if needed), then do
      n number residual block computation
  """

  act: str = 'gelu_tanh'
  norm: str = 'layer'
  bias: bool = True
  winit: str | Callable = nn.Initializer('trunc_normal')
  binit: str | Callable = nn.Initializer('zeros')
  outscale: float = 1.0

  def __init__(self, units: int, stride: int):
    self.units = units
    self.ckw = {k: getattr(self, k) for k in ('bias', 'winit', 'binit')}
    self.outkw = {k: getattr(self, k) for k in (*self.ckw, 'outscale')}
    self._act = nn.get_act(self.act)
    self.stride = stride

  def __call__(self, x: jax.Array) -> jax.Array:
    B, H, W, C = x.shape
    residual = x
    if C != self.units * 4 or self.stride != 1: # we need projection to appropriate dim and dimension
      residual = self.sub(f'inproj', nn.Conv2D, self.units * 4, 1, self.stride, **self.ckw)(residual) # ViT does not use bias
      residual = self.sub('normin', nn.Norm, self.norm)(residual)
    # print(f'[ResidualBlock] 1: {x.shape}')
    x = self.sub(f'cnn0', nn.Conv2D, self.units, 1, 1, **self.ckw)(x)
    x = self.sub(f'norm0', nn.Norm, self.norm)(x)
    x = self._act(x)
    x = self.sub(f"cnn1", nn.Conv2D, self.units, 3, self.stride, **self.ckw)(x)
    x = self.sub(f'norm1', nn.Norm, self.norm)(x)
    x = self._act(x)
    # print(f'[ResidualBlock] 2: {x.shape}')
    x = self.sub(f"outproj", nn.Conv2D, self.units * 4, 1, 1, **self.outkw)(x)
    x = self.sub(f"norm2", nn.Norm, self.norm)(x)
    # print(f'[ResidualBlock] 3: {x.shape}')
    return self._act(x + residual)



class VisionTransformer(nn.Module):

  """Implementation of VisionTransformer

  Args:
    mlp_units (int): Default to 1024.
    cnn_units (int): Default to 64.
    layers (int): Default to 12.
    stage (tuple): Default to (1,). How many blocks of Resnet per resnet stage
      (down sampling stage) (if using resnet as encoder)
    patch (int): Default to 8.
    heads (int): Default to 8.
    ffup (int): Default to 4.
    act (str): Default to 'silu'.
    norm (str): Default to 'rms'.
    qknorm (str): Default to 'none'.
    dropout (float): Default to 0.0.
    bias (bool): Default to True.
    winit (str | Callable): Default to nn.Initializer('trunc_normal').
    binit (str | Callable): Default to nn.Initializer('zeros').
    outscale (float): Default to 1.0.
    use_resnet (bool): Default to True. Whether to use resnet as the first layer.
      This is the implementation of hybrid ViT (also mentioned in the paper as BiT).
  """

  mlp_units: int = 1024
  cnn_units: int = 64
  layers: int = 12
  stage: tuple = (1,)
  patch: int = 8
  heads: int = 8
  ffup: int = 4
  act: str = 'gelu'
  norm: str = 'layer'
  qknorm: str = 'none'
  dropout: float = 0.0
  bias: bool = True
  winit: str | Callable = nn.Initializer('trunc_normal')
  binit: str | Callable = nn.Initializer('zeros')
  outscale: float = 1.0
  use_resnet: bool = True

  def __init__(self):
    self._act = nn.get_act(self.act)
    self.lkw = {k: getattr(self, k) for k in ('bias', 'winit', 'binit')}
    self.akw = {k: getattr(self, k) for k in ('heads', 'qknorm', 'outscale', 'dropout')}
    self.ckw = {k: getattr(self, k) for k in ('bias', 'winit', 'binit')}
    self.rkw = {k: getattr(self, k) for k in ('bias', 'winit', 'binit', 'norm', 'act', 'outscale')}

  def __call__(self, x: jax.Array, masks: list[jax.Array] | jax.Array | None = None, training=True):
    """Perform forward pass of VisionTransformer.

    Args:
        x (jax.Array): Input image/video of shape (B, H, W, C)
        masks (list[jax.Array] | jax.Array | None, optional): List of masks
          of shape (B, K) or single mask of shape (B, K). Defaults to None.
          This mask will be applied to the patches of the input image/video.
        training (bool, optional): Whether to run in training mode. Defaults to True.

    Returns:
        jax.Array: Output of shape (B, K, C). K is the (masked) number of patches
    """
    if masks is not None and not isinstance(masks, list):
      masks = [masks]
    assert (self.use_resnet and masks is None) or not self.use_resnet, 'masking patches is not supported for hybrid ViT architecture.'
    B, H, W, C = x.shape
    # Image encoding ---------------------------------
    # ViT use this
    if self.use_resnet:
      x = self.sub("in", nn.Conv2D, self.cnn_units, 7, 2, pad='valid', **self.ckw)(x) # ViT does not use bias
      x = self.sub('normin', nn.Norm, self.norm)(x)
      x = self._act(x)
      x = flax.linen.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
      cnn_units = self.cnn_units
      for stage, block in enumerate(self.stage):
        with nn.scope(f"stage{stage}"):
          for b in range(block):
            # we do not down scale the whole first stage
            # and we always downscale for the first block
            down = (stage != 0 and b == 0)
            stride = 2 if down else 1
            x = self.sub(f"block{b}", ResidualBlock, cnn_units, stride, **self.rkw)(x)
            # print(f'[ViT] {stage}-{b}: {x.shape}')
            # print("--------------------------------")
        cnn_units *= 2
    patch = self.sub("patch", nn.Conv2D, self.mlp_units, self.patch, self.patch,
      pad="valid", **self.ckw)(x) # ViT does not use bias
    # ------------------------------------------------
    _, h, w, c = patch.shape
    patch = patch.reshape((B, h*w, c)) # (B, HW, C)
    pos = jnp.arange(h*w) # (HW,)
    x = patch + self.sub("pos", nn.Embedding, h*w, self.mlp_units)(pos)[None] # (B, HW, C) + (1, HW, C) -> (B, HW, C)
    if masks is not None:
      xs = take_patches_by_masks(x, masks) # [(B, K ,C)]
      x = jnp.concatenate(xs, axis=0) # (sum(B), K, C)
    x = nn.cast(x) # (B, HW, E)
    # Normal Transformer (same code as Transformer) ---------------
    for i in range(self.layers):
      with nn.scope(f'layer{i}'):
        skip = x
        x = self.sub('norm0', nn.Norm, self.norm)(x)
        x  = self.sub('mha', nn.Attention, self.mlp_units, **self.lkw, **self.akw)(x, x, None, training=training)
        x += skip
        skip = x
        x = self.sub('norm1', nn.Norm, self.norm)(x)
        ff1 = self.sub('ff0', nn.Linear, self.mlp_units * self.ffup, **self.lkw)
        ff2 = self.sub('ff1', nn.Linear, self.mlp_units, **self.lkw, outscale=self.outscale)
        x = ff2(self._act(ff1(x)))
        x += skip
    # -------------------------------------------------------------
    x = self.sub('outnorm', nn.Norm, self.norm)(x)
    return x



