"""
File: networks.py
Author: Viet Nguyen
Date: 2025-05-23

Description: This contain the all networks, adapted from Dreamerv3.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Dict
import math

from lib import nn
from lib.common import Space
from lib.modeling.transformers import VisionTransformer, Transformer
from lib.nlp.masking import (
  create_3d_causal_attention_mask,
  create_3d_attention_mask_from_2d_positive_mask,
  create_3d_attention_mask_from_2d_positive_mask_rows_columns,
  combine_masks
)

class Encoder(nn.Module):
  """
  A flexible neural network encoder for processing multi-modal observations.

  This encoder can handle both vector and image-based inputs, applying different
  processing strategies based on input type. It supports:
  - Vector inputs processed through multi-layer perceptron (MLP)
  - Image inputs processed through convolutional neural network (CNN)
  - Optional symlog transformation for continuous inputs
  - Configurable network depth, units, normalization, and activation

  Example config:
    # for 192x192 input
    **{depth: 64, mults: [2, 3, 4, 4, 4, 4], layers: 3, units: 1024, act: silu, norm: rms, winit: trunc_normal_in, symlog: True, outer: False, kernel: 5, strided: False}
  """

  units: int = 1024  # Number of units in the MLP layers
  norm: str = 'rms'  # Normalization type (Root Mean Square)
  act: str = 'gelu'  # Activation function (Gaussian Error Linear Unit)
  depth: int = 64  # Base depth for convolutional layers
  mults: tuple = (2, 3, 4, 4, 4)  # Multipliers for increasing depth in CNN layers
  layers: int = 3  # Number of MLP layers
  kernel: int = 5  # Kernel size for convolutional layers
  symlog: bool = True  # Whether to apply symlog transformation to continuous inputs
  outer: bool = False  # Special handling for first CNN layer
  strided: bool = False  # Whether to use strided convolutions

  def __init__(self, obs_space, **kw):
    """
    Initialize the encoder with observation space and additional configuration.

    Args:
      obs_space (dict): Dictionary of observation spaces defining input characteristics
      **kw: Additional keyword arguments for network configuration
    """
    assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
    self.obs_space = obs_space
    self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
    self.depths = tuple(self.depth * mult for mult in self.mults)
    self.kw = kw

  def __call__(self, obs, reset, training, single=False):
    """
    Process multi-modal observations through vector and image encoders.

    Args:
      obs (dict): Input observations dictionary
      reset (jax.Array): Reset signal for handling episode boundaries
      training (bool): Training mode flag
      single (bool, optional): Whether processing a single timestep. Defaults to False.

    Returns:
      embed (jax.Array): Encoded representation of input observations
    """
    bdims = 1 if single else 2
    outs = []
    bshape = reset.shape

    if self.veckeys:
      vspace = {k: self.obs_space[k] for k in self.veckeys}
      vecs = {k: obs[k] for k in self.veckeys}
      squish = nn.functional.symlog if self.symlog else lambda x: x
      x = nn.DictConcat(vspace, 1, squish=squish)(vecs)
      x = x.reshape((-1, *x.shape[bdims:]))
      for i in range(self.layers):
        x = self.sub(f'mlp{i}', nn.Linear, self.units, **self.kw)(x)
        x = nn.get_act(self.act)(self.sub(f'mlp{i}norm', nn.Norm, self.norm)(x))
      outs.append(x)

    if self.imgkeys:
      K = self.kernel
      imgs = [obs[k] for k in sorted(self.imgkeys)]
      assert all(x.dtype == jnp.uint8 for x in imgs)
      x = nn.cast(jnp.concatenate(imgs, -1), force=True) / 255 - 0.5
      x = x.reshape((-1, *x.shape[bdims:]))
      for i, depth in enumerate(self.depths):
        if self.outer and i == 0:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
        elif self.strided:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, 2, **self.kw)(x)
        else:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
          B, H, W, C = x.shape
          x = x.reshape((B, H // 2, 2, W // 2, 2, C)).max((2, 4))
        x = nn.get_act(self.act)(self.sub(f'cnn{i}norm', nn.Norm, self.norm)(x))
      assert 3 <= x.shape[-3] <= 16, x.shape
      assert 3 <= x.shape[-2] <= 16, x.shape
      x = x.reshape((x.shape[0], -1))
      outs.append(x)

    x = jnp.concatenate(outs, -1)
    embed = x.reshape((*bshape, *x.shape[1:]))
    return embed


class EncoderViT(nn.Module):
  """
  A flexible neural network encoder for processing multi-modal observations.

  This encoder can handle both vector and image-based inputs, applying different
  processing strategies based on input type. It supports:
  - Vector inputs processed through multi-layer perceptron (MLP)
  - Image inputs processed through convolutional neural network (CNN)
  - Optional symlog transformation for continuous inputs
  - Configurable network depth, units, normalization, and activation

  Example config:
    **{depth: 64, mults: [2, 3, 4, 4], layers: 3, units: 1024, act: silu, norm: rms, winit: trunc_normal_in, symlog: True, outer: False, kernel: 5, strided: False}
  """

  units: int = 768  # Number of units in the MLP layers
  norm: str = 'rms'  # Normalization type (Root Mean Square)
  act: str = 'gelu'  # Activation function (Gaussian Error Linear Unit)
  depth: int = 64  # Base depth for convolutional layers
  # mults: tuple = (2, 3, 4, 4)  # Multipliers for increasing depth in CNN layers
  layers: int = 3  # Number of MLP layers
  kernel: int = 5  # Kernel size for convolutional layers
  symlog: bool = True  # Whether to apply symlog transformation to continuous inputs
  outer: bool = False  # Special handling for first CNN layer
  strided: bool = False  # Whether to use strided convolutions
  # ViT config
  vit_mlp_layers: int = 4
  stage: tuple = (1,)
  patch: int = 16
  heads: int = 4
  ffup: int = 4
  qknorm: str = 'none'
  dropout: float = 0.2
  use_resnet: bool = False

  def __init__(self, obs_space, **kw):
    """
    Initialize the encoder with observation space and additional configuration.

    Args:
      obs_space (dict): Dictionary of observation spaces defining input characteristics
      **kw: Additional keyword arguments for network configuration
    """
    assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
    self.obs_space = obs_space
    self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
    # self.depths = tuple(self.depth * mult for mult in self.mults)
    self.kw = kw

  def __call__(self, obs, reset, training, single=False):
    """
    Process multi-modal observations through vector and image encoders.

    Args:
      obs (dict): Input observations dictionary
      reset (jax.Array): Reset signal for handling episode boundaries
      training (bool): Training mode flag
      single (bool, optional): Whether processing a single timestep. Defaults to False.

    Returns:
      embed (jax.Array): Encoded representation of input observations
    """
    bdims = 1 if single else 2
    outs = []
    bshape = reset.shape

    if self.veckeys:
      vspace = {k: self.obs_space[k] for k in self.veckeys}
      vecs = {k: obs[k] for k in self.veckeys}
      squish = nn.functional.symlog if self.symlog else lambda x: x
      x = nn.DictConcat(vspace, 1, squish=squish)(vecs)
      x = x.reshape((-1, *x.shape[bdims:]))
      for i in range(self.layers):
        x = self.sub(f'mlp{i}', nn.Linear, self.units, **self.kw)(x)
        x = nn.get_act(self.act)(self.sub(f'mlp{i}norm', nn.Norm, self.norm)(x))
      outs.append(x)

    if self.imgkeys:
      K = self.kernel
      imgs = [obs[k] for k in sorted(self.imgkeys)]
      assert all(x.dtype == jnp.uint8 for x in imgs)
      x = nn.cast(jnp.concatenate(imgs, -1), force=True) / 255 - 0.5
      x = x.reshape((-1, *x.shape[bdims:]))
      x = self.sub("vit", VisionTransformer, mlp_units = self.units,
        cnn_units = self.depth, layers = self.vit_mlp_layers, stage = self.stage,
        patch = self.patch, heads = self.heads, ffup = self.ffup,
        act = self.act, norm = self.norm, qknorm = self.qknorm,
        dropout = self.dropout, use_resnet = self.use_resnet)(x, training=training)
      # x = self.sub("pool", nn.AttentiveProbePooling, hidden=self.units, heads = self.heads,
      #   dropout = self.dropout, qknorm = self.qknorm, act = self.act,
      #   norm = self.norm)(x, training=training)
      assert 3 <= x.shape[-3] <= 16, x.shape
      assert 3 <= x.shape[-2] <= 16, x.shape
      x = x.reshape((x.shape[0], -1))
      outs.append(x)

    x = jnp.concatenate(outs, -1)
    embed = x.reshape((*bshape, *x.shape[1:]))
    return embed


class SimpleDecoder(nn.Module):
  """
  This one take in a single vector instead of a dict of vectors
  """

  units: int = 1024
  norm: str = 'rms'
  act: str = 'gelu'
  outscale: float = 1.0
  depth: int = 64
  mults: tuple = (2, 3, 4, 4)
  layers: int = 3
  kernel: int = 5
  symlog: bool = True
  bspace: int = 8
  outer: bool = False
  strided: bool = False

  def __init__(self, obs_space, **kw):
    """
    Initialize the decoder with observation space and additional configuration.

    Args:
      obs_space (dict): Dictionary of observation spaces defining output characteristics
      **kw: Additional keyword arguments for network configuration
    """
    assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
    self.obs_space = obs_space
    self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
    self.depths = tuple(self.depth * mult for mult in self.mults)
    self.imgdep = sum(obs_space[k].shape[-1] for k in self.imgkeys)
    self.imgres = self.imgkeys and obs_space[self.imgkeys[0]].shape[:-1]
    self.kw = kw

  def __call__(self, feat: jax.Array, reset: jax.Array, training: bool):
    """
    Reconstruct multi-modal observations from latent features.

    Args:
      feat (dict): Dictionary of latent features, typically containing 'stoch' and 'deter' keys
      reset (jax.Array): Reset signal for handling episode boundaries
      training (bool): Training mode flag

    Returns:
      recons (dict): Dictionary of reconstructed observations for each output type
    """

    K = self.kernel
    recons = {}
    bshape = reset.shape
    inp = nn.cast(feat)
    inp = inp.reshape((math.prod(bshape), -1))

    if self.veckeys:
      spaces = {k: self.obs_space[k] for k in self.veckeys}
      o1, o2 = 'categorical', ('symlog_mse' if self.symlog else 'mse')
      outputs = {k: o1 if v.discrete else o2 for k, v in spaces.items()}
      kw = dict(**self.kw, act=self.act, norm=self.norm)
      x = self.sub('mlp', nn.MLP, self.layers, self.units, **kw)(inp)
      x = x.reshape((*bshape, *x.shape[1:]))
      kw = dict(**self.kw, outscale=self.outscale)
      outs = self.sub('vec', nn.DictHead, spaces, outputs, **kw)(x)
      recons.update(outs)

    if self.imgkeys:
      factor = 2 ** (len(self.depths) - int(bool(self.outer)))
      minres = [int(x // factor) for x in self.imgres]
      assert 3 <= minres[0] <= 16, minres
      assert 3 <= minres[1] <= 16, minres
      shape = (*minres, self.depths[-1])
      if self.bspace:
        u, g = math.prod(shape), self.bspace
        x = nn.cast(feat)
        x = x.reshape((-1, x.shape[-1]))
        x0 = self.sub('sp0', nn.BlockLinear, u, g, **self.kw)(x) # (..., dim)
        x0 = x0.reshape((*x0.shape[:-1], g, minres[0], minres[1], -1)) # (..., dim) -> (..., g, h, w, c)
        # # (..., g, h, w, c) -> (..., h, w, g, c) -> (..., h, w, gc)
        s = len(x0.shape) - 4
        x0 = x0.transpose((*range(s), s + 1, s + 2, s, s + 3)).reshape((*x0.shape[:-4], minres[0], minres[1], -1))
        x1 = self.sub('sp1', nn.Linear, 2 * self.units, **self.kw)(x)
        x1 = nn.get_act(self.act)(self.sub('sp1norm', nn.Norm, self.norm)(x1))
        x1 = self.sub('sp2', nn.Linear, shape, **self.kw)(x1)
        x = nn.get_act(self.act)(self.sub('spnorm', nn.Norm, self.norm)(x0 + x1))
      else:
        x = self.sub('space', nn.Linear, shape, **kw)(inp)
        x = nn.get_act(self.act)(self.sub('spacenorm', nn.Norm, self.norm)(x))
      for i, depth in reversed(list(enumerate(self.depths[:-1]))):
        if self.strided:
          kw = dict(**self.kw, transp=True)
          x = self.sub(f'conv{i}', nn.Conv2D, depth, K, 2, **kw)(x)
        else:
          x = x.repeat(2, -2).repeat(2, -3)
          x = self.sub(f'conv{i}', nn.Conv2D, depth, K, **self.kw)(x)
        x = nn.get_act(self.act)(self.sub(f'conv{i}norm', nn.Norm, self.norm)(x))
      if self.outer:
        kw = dict(**self.kw, outscale=self.outscale)
        x = self.sub('imgout', nn.Conv2D, self.imgdep, K, **kw)(x)
      elif self.strided:
        kw = dict(**self.kw, outscale=self.outscale, transp=True)
        x = self.sub('imgout', nn.Conv2D, self.imgdep, K, 2, **kw)(x)
      else:
        x = x.repeat(2, -2).repeat(2, -3)
        kw = dict(**self.kw, outscale=self.outscale)
        x = self.sub('imgout', nn.Conv2D, self.imgdep, K, **kw)(x)
      x = jax.nn.sigmoid(x)
      x = x.reshape((*bshape, *x.shape[1:]))
      split = np.cumsum(
          [self.obs_space[k].shape[-1] for k in self.imgkeys][:-1])
      for k, out in zip(self.imgkeys, jnp.split(x, split, -1)):
        out = nn.distributions.Agg(nn.distributions.MSE(out), 3, jnp.sum)
        recons[k] = out
    return recons


class Decoder(nn.Module):
  """
  A flexible neural network decoder for reconstructing multi-modal observations.

  This decoder can handle both vector and image-based outputs, applying different
  reconstruction strategies based on output type. It supports:
  - Vector outputs processed through multi-layer perceptron (MLP)
  - Image outputs processed through transposed convolutional neural network (CNN)
  - Optional symlog transformation for continuous outputs
  - Configurable network depth, units, normalization, and activation

  Attributes:
    units (int): Number of units in the MLP layers. Defaults to 1024.
    norm (str): Normalization type. Defaults to 'rms'.
    act (str): Activation function. Defaults to 'gelu'.
    outscale (float): Output scaling factor. Defaults to 1.0.
    depth (int): Base depth for convolutional layers. Defaults to 64.
    mults (tuple): Multipliers for increasing depth in CNN layers. Defaults to (2, 3, 4, 4).
    layers (int): Number of MLP layers. Defaults to 3.
    kernel (int): Kernel size for convolutional layers. Defaults to 5.
    symlog (bool): Whether to apply symlog transformation to continuous outputs. Defaults to True.
    bspace (int): Block space size for spatial transformations. Defaults to 8.
    outer (bool): Special handling for first CNN layer. Defaults to False.
    strided (bool): Whether to use strided convolutions. Defaults to False.
  """

  units: int = 1024
  norm: str = 'rms'
  act: str = 'gelu'
  outscale: float = 1.0
  depth: int = 64
  mults: tuple = (2, 3, 4, 4)
  layers: int = 3
  kernel: int = 5
  symlog: bool = True
  bspace: int = 8
  outer: bool = False
  strided: bool = False

  def __init__(self, obs_space, **kw):
    """
    Initialize the decoder with observation space and additional configuration.

    Args:
      obs_space (dict): Dictionary of observation spaces defining output characteristics
      **kw: Additional keyword arguments for network configuration
    """
    assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
    self.obs_space = obs_space
    self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
    self.depths = tuple(self.depth * mult for mult in self.mults)
    self.imgdep = sum(obs_space[k].shape[-1] for k in self.imgkeys)
    self.imgres = self.imgkeys and obs_space[self.imgkeys[0]].shape[:-1]
    self.kw = kw

  def __call__(self, feat: Dict[str, jax.Array], reset: jax.Array, training: bool):
    """
    Reconstruct multi-modal observations from latent features.

    Args:
      feat (dict): Dictionary of latent features, typically containing 'stoch' and 'deter' keys
      reset (jax.Array): Reset signal for handling episode boundaries
      training (bool): Training mode flag

    Returns:
      recons (dict): Dictionary of reconstructed observations for each output type
    """
    assert feat['deter'].shape[-1] % self.bspace == 0
    K = self.kernel
    recons = {}
    bshape = reset.shape
    inp = [nn.cast(feat[k]) for k in ('stoch', 'deter')]
    inp = [x.reshape((math.prod(bshape), -1)) for x in inp]
    inp = jnp.concatenate(inp, -1)

    if self.veckeys:
      spaces = {k: self.obs_space[k] for k in self.veckeys}
      o1, o2 = 'categorical', ('symlog_mse' if self.symlog else 'mse')
      outputs = {k: o1 if v.discrete else o2 for k, v in spaces.items()}
      kw = dict(**self.kw, act=self.act, norm=self.norm)
      x = self.sub('mlp', nn.MLP, self.layers, self.units, **kw)(inp)
      x = x.reshape((*bshape, *x.shape[1:]))
      kw = dict(**self.kw, outscale=self.outscale)
      outs = self.sub('vec', nn.DictHead, spaces, outputs, **kw)(x)
      recons.update(outs)

    if self.imgkeys:
      factor = 2 ** (len(self.depths) - int(bool(self.outer)))
      minres = [int(x // factor) for x in self.imgres]
      assert 3 <= minres[0] <= 16, minres
      assert 3 <= minres[1] <= 16, minres
      shape = (*minres, self.depths[-1])
      if self.bspace:
        u, g = math.prod(shape), self.bspace
        x0, x1 = nn.cast((feat['deter'], feat['stoch']))
        x1 = x1.reshape((*x1.shape[:-2], -1))
        x0 = x0.reshape((-1, x0.shape[-1]))
        x1 = x1.reshape((-1, x1.shape[-1]))
        x0 = self.sub('sp0', nn.BlockLinear, u, g, **self.kw)(x0) # (..., dim)
        x0 = x0.reshape((*x0.shape[:-1], g, minres[0], minres[1], -1)) # (..., dim) -> (..., g, h, w, c)
        # # (..., g, h, w, c) -> (..., h, w, g, c) -> (..., h, w, gc)
        s = len(x0.shape) - 4
        x0 = x0.transpose((*range(s), s + 1, s + 2, s, s + 3)).reshape((*x0.shape[:-4], minres[0], minres[1], -1))
        x1 = self.sub('sp1', nn.Linear, 2 * self.units, **self.kw)(x1)
        x1 = nn.get_act(self.act)(self.sub('sp1norm', nn.Norm, self.norm)(x1))
        x1 = self.sub('sp2', nn.Linear, shape, **self.kw)(x1)
        x = nn.get_act(self.act)(self.sub('spnorm', nn.Norm, self.norm)(x0 + x1))
      else:
        x = self.sub('space', nn.Linear, shape, **kw)(inp)
        x = nn.get_act(self.act)(self.sub('spacenorm', nn.Norm, self.norm)(x))
      for i, depth in reversed(list(enumerate(self.depths[:-1]))):
        if self.strided:
          kw = dict(**self.kw, transp=True)
          x = self.sub(f'conv{i}', nn.Conv2D, depth, K, 2, **kw)(x)
        else:
          x = x.repeat(2, -2).repeat(2, -3)
          x = self.sub(f'conv{i}', nn.Conv2D, depth, K, **self.kw)(x)
        x = nn.get_act(self.act)(self.sub(f'conv{i}norm', nn.Norm, self.norm)(x))
      if self.outer:
        kw = dict(**self.kw, outscale=self.outscale)
        x = self.sub('imgout', nn.Conv2D, self.imgdep, K, **kw)(x)
      elif self.strided:
        kw = dict(**self.kw, outscale=self.outscale, transp=True)
        x = self.sub('imgout', nn.Conv2D, self.imgdep, K, 2, **kw)(x)
      else:
        x = x.repeat(2, -2).repeat(2, -3)
        kw = dict(**self.kw, outscale=self.outscale)
        x = self.sub('imgout', nn.Conv2D, self.imgdep, K, **kw)(x)
      x = jax.nn.sigmoid(x)
      x = x.reshape((*bshape, *x.shape[1:]))
      split = np.cumsum(
          [self.obs_space[k].shape[-1] for k in self.imgkeys][:-1])
      for k, out in zip(self.imgkeys, jnp.split(x, split, -1)):
        out = nn.distributions.Agg(nn.distributions.MSE(out), 3, jnp.sum)
        recons[k] = out
    return recons


class SimpleLanguageModel(nn.Module):

  units: int = 256
  layers: int = 4
  heads: int = 8
  ffup: int = 4
  act: str = 'silu'
  norm: str = 'rms'
  qknorm: str = 'none'
  dropout: float = 0.0
  bias: bool = True
  winit: str | Callable = nn.Initializer('trunc_normal')
  binit: str | Callable = nn.Initializer('zeros')
  outscale: float = 1.0
  pool_depth: int = 1

  def __init__(self, vocab_size: int, max_sequence_length: int):
    self._word_embedding = nn.Embedding(vocab_size, self.units, name="word_embedding")
    self._position_embedding = nn.Embedding(max_sequence_length, self.units, name="position_embedding")
    self._norm1 = nn.Norm(self.norm, name="norm1")
    self._ln_out = nn.Norm(self.norm, name="ln_out")
    self._transformer = Transformer(units = self.units, layers = self.layers, heads = self.heads, ffup = self.ffup,
      act = self.act, norm = self.norm, qknorm = self.qknorm, dropout = self.dropout, bias = self.bias,
      winit = self.winit, binit = self.binit, outscale = self.outscale, name = "transformer")
    self._pooler = nn.AttentiveProbePooling(self.units, depth = self.pool_depth, heads = self.heads,
      dropout = self.dropout, qknorm = self.qknorm, act = self.act, norm = self.norm,
      bias = self.bias, winit = self.winit, binit = self.binit, outscale = self.outscale, name = "pooler")

  def __call__(self, input_ids: jax.Array, attention_mask: jax.Array, single=False, training=True) -> jax.Array:
    bdims = 1 if single else 2
    bshape = input_ids.shape[:bdims]
    input_ids = input_ids.reshape((-1, *input_ids.shape[bdims:]))
    attention_mask = attention_mask.reshape((-1, *attention_mask.shape[bdims:]))

    proc_attention_mask = combine_masks(
      create_3d_attention_mask_from_2d_positive_mask_rows_columns(attention_mask),
      create_3d_causal_attention_mask(attention_mask.shape[0], attention_mask.shape[1])
    ) # (B, seqlen, seqlen)

    # Encode text
    word_embedding = self._word_embedding(input_ids) # (B, seqlen, dim)
    position_embedding = self._position_embedding(jnp.arange(word_embedding.shape[1])) # (seqlen, dim)
    text_features = word_embedding + position_embedding[None] # (B, seqlen, dim)
    text_features = self._transformer(text_features, mask=proc_attention_mask, training=training) # (B, seqlen, dim)
    text_features = self._norm1(text_features)
    text_features = self._pooler(text_features, training=training) # (B, dim)
    text_features = self._ln_out(text_features) # (B, dim)
    text_features = text_features.reshape((*bshape, *text_features.shape[1:])) # (*B, dim)
    return text_features


class QueryEncoder(nn.Module):
  """
  A flexible neural network encoder for processing multi-modal goal queries.

  This encoder can handle both vector and image-based inputs, applying different
  processing strategies based on input type. It supports:
  - Vector inputs processed through multi-layer perceptron (MLP)
  - Image inputs processed through convolutional neural network (CNN)
  - Optional symlog transformation for continuous inputs
  - Configurable network depth, units, normalization, and activation
  - Additional language model for query encoding

  Example config:
    # for 192x192 input
    **{depth: 64, mults: [2, 3, 4, 4, 4, 4], layers: 3, units: 1024, act: silu, norm: rms, winit: trunc_normal_in, symlog: True, outer: False, kernel: 5, strided: False}
  """

  units: int = 1024  # Number of units in the MLP layers
  norm: str = 'rms'  # Normalization type (Root Mean Square)
  act: str = 'gelu'  # Activation function (Gaussian Error Linear Unit)
  depth: int = 64  # Base depth for convolutional layers
  mults: tuple = (2, 3, 4, 4, 4)  # Multipliers for increasing depth in CNN layers
  layers: int = 3  # Number of MLP layers
  kernel: int = 5  # Kernel size for convolutional layers
  symlog: bool = True  # Whether to apply symlog transformation to continuous inputs
  outer: bool = False  # Special handling for first CNN layer
  strided: bool = False  # Whether to use strided convolutions
  learnablegoal: bool = True

  # Language model
  lang_units: int = 256
  lang_layers: int = 12
  lang_heads: int = 8
  lang_ffup: int = 4
  lang_act: str = 'silu'
  lang_norm: str = 'rms'
  lang_qknorm: str = 'none'
  lang_dropout: float = 0.0
  lang_pool_depth: int = 1

  def __init__(self, obs_space, vocab_size: int | None = None, max_sequence_length: int | None = None, **kw):
    """
    Initialize the encoder with observation space and additional configuration.

    Args:
      obs_space (dict): Dictionary of observation spaces defining input characteristics
      **kw: Additional keyword arguments for network configuration
    """
    assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
    self.obs_space = obs_space
    self.langkeys = [k for k, s in obs_space.items() if (len(s.shape) == 1 and s.dtype == jnp.int32)]
    self.maskkeys = [k for k, s in obs_space.items() if (len(s.shape) == 1 and s.dtype == jnp.bool)]
    assert (len(self.langkeys) == 0 and len(self.maskkeys) == 0) or \
      (len(self.langkeys) == 1 and len(self.maskkeys) == 1 and vocab_size\
        is not None and max_sequence_length is not None), (self.langkeys,\
          self.maskkeys, vocab_size, max_sequence_length)
    self.vocab_size = vocab_size
    self.max_sequence_length = max_sequence_length
    self.veckeys = [k for k, s in obs_space.items() if (len(s.shape) <= 2 and k not in self.langkeys and k not in self.maskkeys)]
    self.imgkeys = [k for k, s in obs_space.items() if (len(s.shape) == 3 and k not in self.langkeys and k not in self.maskkeys)]
    self.depths = tuple(self.depth * mult for mult in self.mults)
    self.kw = kw
    self.lang_kw = {f"{k[5:]}": getattr(self, k) for k in ['lang_units', 'lang_layers',
      'lang_heads', 'lang_ffup', 'lang_act', 'lang_norm', 'lang_qknorm',
      'lang_dropout', 'lang_pool_depth']}

  def __call__(self, obs, reset, training, single=False):
    """
    Process multi-modal observations through vector and image encoders.

    Args:
      obs (dict): Input observations dictionary
      reset (jax.Array): Reset signal for handling episode boundaries
      training (bool): Training mode flag
      single (bool, optional): Whether processing a single timestep. Defaults to False.

    Returns:
      embed (jax.Array): Encoded representation of input observations
    """
    bdims = 1 if single else 2
    outs = []
    bshape = reset.shape

    # in case there exists no fields, replace it with zeros
    # outs.append(jnp.zeros((np.prod(bshape), 1)))
    if self.learnablegoal:
      learnable_goal = self.value('learnablegoal', jnp.zeros, (1, self.units))
      learnable_goal = learnable_goal.repeat(np.prod(bshape), axis=0)
      outs.append(learnable_goal)
    else:
      padding = self.value('padding', jnp.zeros, (1, 1)) * 0
      padding = padding.repeat(np.prod(bshape), axis=0)
      outs.append(padding)

    if self.veckeys:
      vspace = {k: self.obs_space[k] for k in self.veckeys}
      vecs = {k: obs[k] for k in self.veckeys}
      squish = nn.functional.symlog if self.symlog else lambda x: x
      x = nn.DictConcat(vspace, 1, squish=squish)(vecs)
      x = x.reshape((-1, *x.shape[bdims:]))
      for i in range(self.layers):
        x = self.sub(f'mlp{i}', nn.Linear, self.units, **self.kw)(x)
        x = nn.get_act(self.act)(self.sub(f'mlp{i}norm', nn.Norm, self.norm)(x))
      outs.append(x)

    if self.langkeys and self.maskkeys:
      langkey, maskkey = self.langkeys[0], self.maskkeys[0]
      langs = obs[langkey]
      mask = obs[maskkey]
      langs = langs.reshape((-1, *langs.shape[bdims:]))
      mask = mask.reshape((-1, *mask.shape[bdims:]))
      assert mask.dtype == jnp.bool, mask.dtype
      assert langs.dtype == jnp.int32, langs.dtype
      lang_emb = self.sub('lang', SimpleLanguageModel, vocab_size = self.vocab_size,
        max_sequence_length = self.max_sequence_length,
          **self.lang_kw, **self.kw)(langs, mask, single=True, training=training)
      outs.append(lang_emb)

    if self.imgkeys:
      K = self.kernel
      imgs = [obs[k] for k in sorted(self.imgkeys)]
      assert all(x.dtype == jnp.uint8 for x in imgs)
      x = nn.cast(jnp.concatenate(imgs, -1), force=True) / 255 - 0.5
      x = x.reshape((-1, *x.shape[bdims:]))
      for i, depth in enumerate(self.depths):
        if self.outer and i == 0:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
        elif self.strided:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, 2, **self.kw)(x)
        else:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
          B, H, W, C = x.shape
          x = x.reshape((B, H // 2, 2, W // 2, 2, C)).max((2, 4))
        x = nn.get_act(self.act)(self.sub(f'cnn{i}norm', nn.Norm, self.norm)(x))
      assert 3 <= x.shape[-3] <= 16, x.shape
      assert 3 <= x.shape[-2] <= 16, x.shape
      x = x.reshape((x.shape[0], -1))
      outs.append(x)

    x = jnp.concatenate(outs, -1) # (B, (optional)T, dim)
    embed = x.reshape((*bshape, *x.shape[1:]))
    return embed


class MultimodalEncoder(nn.Module):
  """encode observation and language into the same embedding space
  """

  units: int = 768  # Number of units in the MLP layers
  norm: str = 'rms'  # Normalization type (Root Mean Square)
  act: str = 'gelu'  # Activation function (Gaussian Error Linear Unit)
  depth: int = 64  # Base depth for convolutional layers
  # mults: tuple = (2, 3, 4, 4)  # Multipliers for increasing depth in CNN layers
  layers: int = 3  # Number of MLP layers
  kernel: int = 5  # Kernel size for convolutional layers
  symlog: bool = True  # Whether to apply symlog transformation to continuous inputs
  outer: bool = False  # Special handling for first CNN layer
  strided: bool = False  # Whether to use strided convolutions
  # ViT config
  vit_mlp_layers: int = 4
  stage: tuple = (1,)
  patch: int = 16
  heads: int = 4
  ffup: int = 4
  qknorm: str = 'none'
  dropout: float = 0.2
  use_resnet: bool = False
  # Language model
  lang_units: int = 256
  lang_layers: int = 12
  lang_heads: int = 8
  lang_ffup: int = 4
  lang_act: str = 'silu'
  lang_norm: str = 'rms'
  lang_qknorm: str = 'none'
  lang_dropout: float = 0.0
  lang_pool_depth: int = 1

  def __init__(self, obs_space, vocab_size: int | None = None, max_sequence_length: int | None = None, **kw):
    """
    Initialize the encoder with observation space and additional configuration.

    Args:
      obs_space (dict): Dictionary of observation spaces defining input characteristics
      **kw: Additional keyword arguments for network configuration
    """
    assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
    self.obs_space = obs_space
    self.langkeys = [k for k, s in obs_space.items() if (len(s.shape) == 1 and s.dtype == jnp.int32)]
    self.maskkeys = [k for k, s in obs_space.items() if (len(s.shape) == 1 and s.dtype == jnp.bool)]
    assert (len(self.langkeys) == 0 and len(self.maskkeys) == 0) or \
      (len(self.langkeys) == 1 and len(self.maskkeys) == 1 and vocab_size\
        is not None and max_sequence_length is not None), (self.langkeys,\
          self.maskkeys, vocab_size, max_sequence_length)
    self.vocab_size = vocab_size
    self.max_sequence_length = max_sequence_length
    self.veckeys = [k for k, s in obs_space.items() if (len(s.shape) <= 2 and k not in self.langkeys and k not in self.maskkeys)]
    self.imgkeys = [k for k, s in obs_space.items() if (len(s.shape) == 3 and k not in self.langkeys and k not in self.maskkeys)]
    self.kw = kw
    self.lang_kw = {f"{k[5:]}": getattr(self, k) for k in ['lang_units', 'lang_layers',
      'lang_heads', 'lang_ffup', 'lang_act', 'lang_norm', 'lang_qknorm',
      'lang_dropout', 'lang_pool_depth']}

  def __call__(self, obs, reset, training, single=False):
    """
    Process multi-modal observations through vector and image encoders.

    Args:
      obs (dict): Input observations dictionary
      reset (jax.Array): Reset signal for handling episode boundaries
      training (bool): Training mode flag
      single (bool, optional): Whether processing a single timestep. Defaults to False.

    Returns:
      embed (jax.Array): Encoded representation of input observations
    """
    bdims = 1 if single else 2
    outs = []
    bshape = reset.shape

    if self.veckeys:
      vspace = {k: self.obs_space[k] for k in self.veckeys}
      vecs = {k: obs[k] for k in self.veckeys}
      squish = nn.functional.symlog if self.symlog else lambda x: x
      x = nn.DictConcat(vspace, 1, squish=squish)(vecs)
      x = x.reshape((-1, *x.shape[bdims:]))
      for i in range(self.layers):
        x = self.sub(f'mlp{i}', nn.Linear, self.units, **self.kw)(x)
        x = nn.get_act(self.act)(self.sub(f'mlp{i}norm', nn.Norm, self.norm)(x))
      # print(f'mlp', x.shape)
      outs.append(x)

    if self.langkeys and self.maskkeys:
      langkey, maskkey = self.langkeys[0], self.maskkeys[0]
      langs = obs[langkey]
      mask = obs[maskkey]
      langs = langs.reshape((-1, *langs.shape[bdims:]))
      mask = mask.reshape((-1, *mask.shape[bdims:]))
      assert mask.dtype == jnp.bool, mask.dtype
      assert langs.dtype == jnp.int32, langs.dtype
      lang_emb = self.sub('lang', SimpleLanguageModel, vocab_size = self.vocab_size,
        max_sequence_length = self.max_sequence_length,
          **self.lang_kw, **self.kw)(langs, mask, single=True, training=training)
      # print(f'lang', lang_emb.shape)
      outs.append(lang_emb)

    if self.imgkeys:
      K = self.kernel
      imgs = [obs[k] for k in sorted(self.imgkeys)]
      assert all(x.dtype == jnp.uint8 for x in imgs)
      x = nn.cast(jnp.concatenate(imgs, -1), force=True) / 255 - 0.5
      x = x.reshape((-1, *x.shape[bdims:]))
      x = self.sub("vit", VisionTransformer, mlp_units = self.units,
        cnn_units = self.depth, layers = self.vit_mlp_layers, stage = self.stage,
        patch = self.patch, heads = self.heads, ffup = self.ffup,
        act = self.act, norm = self.norm, qknorm = self.qknorm,
        dropout = self.dropout, use_resnet = self.use_resnet)(x, training=training)
      x = self.sub("vitpool", nn.AttentiveProbePooling, hidden=self.units, depth = 1, heads = self.heads,
        dropout = self.dropout, qknorm = self.qknorm, act = self.act, norm = self.norm)(x)
      # print(f'vitpool', x.shape)
      outs.append(x)

    x = jnp.concatenate(outs, -1)
    embed = x.reshape((*bshape, *x.shape[1:]))
    return embed


class QMoPSSM(nn.Module):

  deter: int = 4096
  hidden: int = 2048
  stoch: int = 32
  classes: int = 32
  mixture: int = 1
  norm: str = 'rms'
  act: str = 'silu'
  unroll: bool = False
  unimix: float = 0.01
  outscale: float = 1.0
  imglayers: int = 2
  obslayers: int = 1
  dynlayers: int = 1
  absolute: bool = False
  blocks: int = 8
  free_nats: float = 1.0

  def __init__(self, act_space, **kw):
    assert self.deter % self.blocks == 0
    self.act_space = act_space
    self.kw = kw

  @property
  def entry_space(self):
    return dict(
      deter=Space(np.float32, self.deter),
      stoch=Space(np.float32, (self.stoch, self.classes)),
      prefdeter=Space(np.float32, self.deter),
      prefstoch=Space(np.float32, (self.stoch, self.classes))
    )

  def initial(self, bsize):
    carry = nn.cast(dict(
        deter=jnp.zeros([bsize, self.deter], nn.f32),
        stoch=jnp.zeros([bsize, self.stoch, self.classes], nn.f32),
        prefdeter=jnp.zeros([bsize, self.deter], nn.f32),
        prefstoch=jnp.zeros([bsize, self.stoch, self.classes], nn.f32)))
    return carry

  def truncate(self, entries, carry=None):
    assert entries['deter'].ndim == 3, entries['deter'].shape
    carry = jax.tree.map(lambda x: x[:, -1], entries)
    return carry

  def truncate_preference(self, entries, carry=None):
    assert entries['prefdeter'].ndim == 3, entries['prefdeter'].shape
    carry = jax.tree.map(lambda x: x[:, -1], entries)
    return carry

  def starts(self, entries, carry, nlast):
    # Extract the last `nlast` steps from the entries and reshape them
    # This method is used to prepare the starting points for imagination or trajectory rollout
    # It takes the last `nlast` time steps from each batch and flattens them into a single dimension
    # This allows processing multiple starting points in parallel

    # Get the batch size from the first element of the carry dictionary
    B = len(jax.tree.leaves(carry)[0])

    # For each tensor in entries:
    # 1. Take the last `nlast` time steps using [:, -nlast:]
    # 2. Reshape to (B * nlast, *remaining_dimensions)
    # This effectively creates a flattened batch of starting points
    return jax.tree.map(
        lambda x: x[:, -nlast:].reshape((B * nlast, *x.shape[2:])), entries)

  def future_query(self, query: jax.Array, carry: Dict[str, jax.Array], nlast: int, length: int):
    # Extract the last `nlast` steps from the entries and reshape them
    # This method is used to prepare the starting points for imagination or trajectory rollout
    # It takes the last `nlast` time steps from each batch and flattens them into a single dimension
    # This allows processing multiple starting points in parallel

    # Get the batch size from the first element of the carry dictionary
    B = len(jax.tree.leaves(carry)[0])

    # For each tensor in entries:
    # 1. Take the last `nlast` time steps using [:, -nlast:]
    # 2. Reshape to (B * nlast, *remaining_dimensions)
    # This effectively creates a flattened batch of starting points
    starts = query[:, -nlast:].reshape((B * nlast, *query.shape[2:]))
    return starts[:, None].repeat(length, axis=1) # (B*T, length, dim)

  def observe(self, carry, input_emb, action, query_emb, reset, training, single=False):
    """Observe method for the RSSM (Recurrent State Space Model)

    Args:
      carry (dict): Carry state containing 'deter' and 'stoch' tensors from previous timestep
        - deter (jnp.ndarray): Deterministic state of shape [batch_size, deter_dim]
        - stoch (jnp.ndarray): Stochastic state of shape [batch_size, stoch_dim, classes]
      input_emb (jnp.ndarray): Input embedding of shape [batch_size, (optional)time_steps, embedding_dim]
      action (jnp.ndarray): Action tensor of shape [batch_size, (optional)time_steps, action_dim]
      reset (jnp.ndarray): Reset flag tensor of shape [batch_size, (optional)time_steps]
      training (bool): Flag indicating whether in training or inference mode
      single (bool, optional): Whether processing a single timestep or multiple. Defaults to False.

    Returns:
      tuple:
        - carry (dict): Updated carry state for next timestep (*B, dim)
        - entries (dict or tuple): State entries (*B, (optional) T, dim)
        - feat (dict or tuple): State features including deterministic and stochastic components (*B, (optional) T, dim) with features
    """
    # 'input_emb' is an embedding or representation of input data
    # In the context of the RSSM (Recurrent State Space Model), it represents
    # a learned representation of the input from an encoder
    carry, input_emb, query_emb, action = nn.cast((carry, input_emb, query_emb, action))
    if single:
      carry, (entry, feat) = self._observe(
          carry, input_emb, action, query_emb, reset, training)
      return carry, entry, feat
    else:
      unroll = jax.tree.leaves(input_emb)[0].shape[1] if self.unroll else 1
      carry, (entries, feat) = nn.scan(
          lambda carry, inputs: self._observe(
              carry, *inputs, training),
          carry, (input_emb, action, query_emb, reset), unroll=unroll, axis=1)
      return carry, entries, feat

  def _observe(self, carry, input_emb, action, query_emb, reset, training):
    deter, stoch, prefdeter, prefstoch, action, query_emb = nn.functional.mask(
        (carry['deter'], carry['stoch'], carry['prefdeter'], carry['prefstoch'], action, query_emb), ~reset)
    action = nn.DictConcat(self.act_space, 1)(action)
    action = nn.functional.mask(action, ~reset)
    deter = self._core(deter, stoch, action)
    prefdeter = self._prefcore(prefdeter, prefstoch, query_emb)
    input_emb = input_emb.reshape((*deter.shape[:-1], -1))
    # x = input_emb if self.absolute else jnp.concatenate([deter, input_emb], -1)
    x = jnp.concatenate([input_emb, query_emb], -1) if self.absolute else jnp.concatenate([deter, input_emb, query_emb], -1)
    for i in range(self.obslayers):
      x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.get_act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
    xp = jnp.concatenate([input_emb, query_emb], -1) if self.absolute else jnp.concatenate([prefdeter, input_emb, query_emb], -1)
    for i in range(self.obslayers):
      xp = self.sub(f'pobs{i}', nn.Linear, self.hidden, **self.kw)(xp)
      xp = nn.get_act(self.act)(self.sub(f'pobs{i}norm', nn.Norm, self.norm)(xp))
    logit = self._logit('obslogit', x)
    plogit = self._mlogit('pobslogit', xp, query_emb)
    stoch = nn.cast(self._dist(logit).sample(seed=nn.seed()))
    prefstoch = nn.cast(self._dist(plogit).sample(seed=nn.seed()))
    carry = dict(deter=deter, stoch=stoch, prefdeter=prefdeter, prefstoch=prefstoch)
    feat = dict(deter=deter, stoch=stoch, prefdeter=prefdeter, prefstoch=prefstoch, logit=logit, plogit=plogit)
    entry = dict(deter=deter, stoch=stoch, prefdeter=prefdeter, prefstoch=prefstoch)
    assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit, prefdeter, prefstoch, plogit))
    return carry, (entry, feat)

  def imagine(self, carry, policy, query, length, training, single=False):
    if single:
      action = policy(nn.sg(carry)) if callable(policy) else policy
      actemb = nn.DictConcat(self.act_space, 1)(action)
      deter = self._core(carry['deter'], carry['stoch'], actemb)
      prefdeter = self._prefcore(carry['prefdeter'], carry['prefstoch'], query)
      logit = self._prior(deter)
      plogit = self._pprior(prefdeter, query)
      stoch = nn.cast(self._dist(logit).sample(seed=nn.seed()))
      prefstoch = nn.cast(self._dist(plogit).sample(seed=nn.seed()))
      carry = nn.cast(dict(deter=deter, stoch=stoch, prefdeter=prefdeter, prefstoch=prefstoch))
      feat = nn.cast(dict(deter=deter, stoch=stoch, prefdeter=prefdeter, prefstoch=prefstoch, logit=logit, plogit=plogit))
      assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit, prefdeter, prefstoch))
      return carry, (feat, action)
    else:
      unroll = length if self.unroll else 1
      if callable(policy):
        carry, (feat, action) = nn.scan(
            lambda c, q: self.imagine(c, policy, q, 1, training, single=True),
            nn.cast(carry), nn.cast(query), length, unroll=unroll, axis=1)
      else:
        carry, (feat, action) = nn.scan(
            lambda c, inputs: self.imagine(c, *inputs, 1, training, single=True),
            nn.cast(carry), (nn.cast(policy), nn.cast(query)), length, unroll=unroll, axis=1)
      # We can also return all carry entries but it might be expensive.
      # entries = dict(deter=feat['deter'], stoch=feat['stoch'])
      # return carry, entries, feat, action
      return carry, feat, action

  def imagine_preference(self, carry, query, length, training, single=False):
    if single:
      prefdeter = self._prefcore(carry['prefdeter'], carry['prefstoch'], query)
      plogit, _ = self._pprior(prefdeter, query)
      prefstoch = nn.cast(self._dist(plogit).sample(seed=nn.seed()))
      carry = nn.cast(dict(prefdeter=prefdeter, prefstoch=prefstoch))
      feat = nn.cast(dict(prefdeter=prefdeter, prefstoch=prefstoch, plogit=plogit))
      assert all(x.dtype == nn.COMPUTE_DTYPE for x in (prefdeter, prefstoch, plogit))
      return carry, feat
    else:
      unroll = length if self.unroll else 1
      carry, feat = nn.scan(
        lambda c, q: self.imagine_preference(c, q, 1, training, single=True),
        nn.cast(carry), nn.cast(query), length, unroll=unroll, axis=1)
      return carry, feat

  def cosine_sim(self, prefdeter, prefstoch, deter, stoch):
    # shape in (*B, deter) and (*B, stoch, classes). out: (*B,)
    shape = prefstoch.shape[:-2] + (self.stoch * self.classes,)
    x1 = prefstoch.reshape(shape)
    x2 = stoch.reshape(shape)
    x1 = jnp.concatenate([prefdeter, x1], -1)
    x2 = jnp.concatenate([deter, x2], -1)
    # x1 and x2 (B, T, L)
    dot = (x1 * x2).sum(-1)
    denom_x1 = jnp.sqrt((x1 ** 2).sum(-1).clip(1e-8))
    denom_x2 = jnp.sqrt((x2 ** 2).sum(-1).clip(1e-8))
    return dot / (denom_x1 * denom_x2).clip(1e-8)

  def loss(self, carry, input_emb, acts, query_emb, rho, reset, training):
    # acts here is the previous actions
    metrics = {}
    klmask = jnp.sign(rho).clip(0, 1)
    carry, entries, feat = self.observe(carry, input_emb, acts, query_emb, reset, training)
    prior = self._prior(feat['deter'])
    post = feat['logit']
    dyn = self._dist(nn.sg(post)).kl(self._dist(prior))
    rep = self._dist(post).kl(self._dist(nn.sg(prior)))
    prefprior: jax.Array = self._pprior(feat['prefdeter'], query_emb) # logit
    prefpost: jax.Array = feat['plogit'] # logit
    prefdyn = self._dist(nn.sg(prefpost)).kl(self._dist(prefprior)) * klmask
    prefrep = self._dist(prefpost).kl(self._dist(nn.sg(prefprior))) * klmask
    if self.free_nats:
      dyn = jnp.maximum(dyn, self.free_nats)
      rep = jnp.maximum(rep, self.free_nats)
      prefdyn = jnp.maximum(prefdyn, self.free_nats) * klmask
      prefrep = jnp.maximum(prefrep, self.free_nats) * klmask
    losses = {'dyn': dyn, 'rep': rep, 'prefdyn': prefdyn, 'prefrep': prefrep}
    metrics['dyn_ent'] = self._dist(prior).entropy().mean()
    metrics['rep_ent'] = self._dist(post).entropy().mean()
    metrics['prefdyn_ent'] = self._dist(prefprior).entropy().mean()
    metrics['prefrep_ent'] = self._dist(prefprior).entropy().mean()
    # Contrastive loss
    similarity_post = self.cosine_sim(feat['prefdeter'], feat['plogit'], nn.sg(feat['deter']), nn.sg(post)) # (B, T)
    similarity_prior = self.cosine_sim(feat['prefdeter'], prefprior, nn.sg(feat['deter']), nn.sg(post)) # (B, T)
    losses['contrast'] = -rho * (similarity_post + similarity_prior)
    losses['mixture'] = self._mixture_specialization_loss(prefprior)
    return carry, entries, losses, feat, metrics

  def _core(self, deter, stoch, action):
    stoch = stoch.reshape((stoch.shape[0], -1))
    action /= nn.sg(jnp.maximum(1, jnp.abs(action)))
    g = self.blocks
    flat2group = lambda x: jnp.reshape(x, x.shape[:-1] + (g, -1)) # ... (g h) -> ... g h
    group2flat = lambda x: jnp.reshape(x, x.shape[:-2] + (-1,)) # ... g h -> ... (g h)
    x0 = self.sub('dynin0', nn.Linear, self.hidden, **self.kw)(deter)
    x0 = nn.get_act(self.act)(self.sub('dynin0norm', nn.Norm, self.norm)(x0))
    x1 = self.sub('dynin1', nn.Linear, self.hidden, **self.kw)(stoch)
    x1 = nn.get_act(self.act)(self.sub('dynin1norm', nn.Norm, self.norm)(x1))
    x2 = self.sub('dynin2', nn.Linear, self.hidden, **self.kw)(action)
    x2 = nn.get_act(self.act)(self.sub('dynin2norm', nn.Norm, self.norm)(x2))
    x = jnp.concatenate([x0, x1, x2], -1)[..., None, :].repeat(g, -2)
    x = group2flat(jnp.concatenate([flat2group(deter), x], -1))
    for i in range(self.dynlayers):
      x = self.sub(f'dynhid{i}', nn.BlockLinear, self.deter, g, **self.kw)(x)
      x = nn.get_act(self.act)(self.sub(f'dynhid{i}norm', nn.Norm, self.norm)(x))
    x = self.sub('dyngru', nn.BlockLinear, 3 * self.deter, g, **self.kw)(x)
    gates = jnp.split(flat2group(x), 3, -1)
    reset, cand, update = [group2flat(x) for x in gates]
    reset = jax.nn.sigmoid(reset)
    cand = jnp.tanh(reset * cand)
    update = jax.nn.sigmoid(update - 1)
    deter = update * cand + (1 - update) * deter
    return deter

  def _prefcore(self, prefdeter, prefstoch, query_emb):
    # NOTE: Considering attenting the query to the preference using special operation so that it is more efficient
    prefstoch = prefstoch.reshape((prefstoch.shape[0], -1))
    g = self.blocks
    flat2group = lambda x: jnp.reshape(x, x.shape[:-1] + (g, -1)) # ... (g h) -> ... g h
    group2flat = lambda x: jnp.reshape(x, x.shape[:-2] + (-1,)) # ... g h -> ... (g h)
    x0 = self.sub('pdynin0', nn.Linear, self.hidden, **self.kw)(prefdeter)
    x0 = nn.get_act(self.act)(self.sub('pdynin0norm', nn.Norm, self.norm)(x0))
    x1 = self.sub('pdynin1', nn.Linear, self.hidden, **self.kw)(prefstoch)
    x1 = nn.get_act(self.act)(self.sub('pdynin1norm', nn.Norm, self.norm)(x1))
    x2 = self.sub('pdynin2', nn.Linear, self.hidden, **self.kw)(query_emb)
    x2 = nn.get_act(self.act)(self.sub('pdynin2norm', nn.Norm, self.norm)(x2))
    x = jnp.concatenate([x0, x1, x2], -1)[..., None, :].repeat(g, -2)
    x = group2flat(jnp.concatenate([flat2group(prefdeter), x], -1))
    for i in range(self.dynlayers):
      x = self.sub(f'pdynhid{i}', nn.BlockLinear, self.deter, g, **self.kw)(x)
      x = nn.get_act(self.act)(self.sub(f'pdynhid{i}norm', nn.Norm, self.norm)(x))
    x = self.sub('pdyngr', nn.BlockLinear, 3 * self.deter, g, **self.kw)(x)
    gates = jnp.split(flat2group(x), 3, -1)
    reset, cand, update = [group2flat(x) for x in gates]
    reset = jax.nn.sigmoid(reset)
    cand = jnp.tanh(reset * cand)
    update = jax.nn.sigmoid(update - 1)
    prefdeter = update * cand + (1 - update) * prefdeter
    return prefdeter

  def _prior(self, feat):
    x = feat
    for i in range(self.imglayers):
      x = self.sub(f'prior{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.get_act(self.act)(self.sub(f'prior{i}norm', nn.Norm, self.norm)(x))
    return self._logit('priorlogit', x)

  def _pprior(self, feat, query_emb):
    x = feat
    for i in range(self.imglayers):
      x = self.sub(f'pprior{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.get_act(self.act)(self.sub(f'pprior{i}norm', nn.Norm, self.norm)(x))
    return self._mlogit('ppriorlogit', x, query_emb)

  def _logit(self, name, x):
    kw = dict(**self.kw, outscale=self.outscale)
    x = self.sub(name, nn.Linear, self.stoch * self.classes, **kw)(x)
    return x.reshape(x.shape[:-1] + (self.stoch, self.classes))

  def preference_gating(self, x, query_emb):
    # x shape: (..., M, stoch, classes) where M is number of mixtures
    # query_emb shape: (..., query_dim)
    # Project query to mixture logits
    query_logits = self.sub('query_gate', nn.Linear, self.mixture, **self.kw)(nn.cast(query_emb))  # (..., M)
    mixture_weights = jax.nn.softmax(query_logits, axis=-1)  # (..., M)
    # Reshape mixture weights for broadcasting
    mixture_weights = mixture_weights[..., None, None]  # (..., M, 1, 1)
    # Weighted sum of mixtures
    weighted_mixture = jnp.sum(x * mixture_weights, axis=-3)  # (..., stoch, classes)
    return weighted_mixture

  def _mlogit(self, name, x, query_emb):
    kw = dict(**self.kw, outscale=self.outscale)
    x = self.sub(name, nn.Linear, self.mixture * self.stoch * self.classes, **kw)(x)
    x = x.reshape(x.shape[:-1] + (self.mixture, self.stoch, self.classes))
    return self.preference_gating(x, query_emb)

  def _dist(self, logits):
    out = nn.distributions.Agg(nn.distributions.OneHot(logits, self.unimix), 1, jnp.sum)
    return out

  def _mixture_specialization_loss(self, mixture_weights):
    """Compute loss to encourage mixture specialization and prevent mode collapse.

    Args:
        mixture_weights: (..., M) weights for each mixture

    Returns:
        loss: scalar loss value
    """
    # Entropy regularization to prevent mode collapse
    # Higher entropy means more diverse mixture usage
    entropy_loss = -self._dist(mixture_weights).entropy() # Negative because we want to maximize entropy
    # L2 regularization on mixture weights to prevent extreme values
    l2_loss = jnp.sum(mixture_weights ** 2, axis=[-1, -2])
    # Combine losses
    total_loss = entropy_loss + 0.1 * l2_loss
    return total_loss

