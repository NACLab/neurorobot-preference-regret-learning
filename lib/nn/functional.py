"""
File: functional.py
Author: Viet Nguyen
Date: 2024-01-23

Description: Include some functional operations
"""

from typing import List, Tuple, Optional
import jax
import jax.numpy as jnp
import numpy as np
import math
# from tensorflow_probability.substrates import jax as tfp
# tfd = tfp.distributions
from . import ninjax as nj
from .utils import cast_to_compute, sg
f32 = jnp.float32


# def bce(inputs: jax.Array, target: jax.Array, input_type='logit', eps=1e-5):
#   if input_type == 'logit':
#     return - (target * (jax.nn.log_sigmoid(inputs)))
#   elif input_type == 'probs':
#     return
#   else:
#     raise ValueError("`input_type` can only be `logit` or `probs`.")


def triplet_contrastive_loss_with_negative_mask(
    anchor: jax.Array, positive: jax.Array,
    negative: jax.Array, negative_mask: jax.Array,
    margin :float = 1.0,
    eps: float = 1e-8
  ) -> jax.Array:
  """Compute batched all pair contrastive loss
    NOTE: Without the mask, the cumulative reward plot is good
    NOTE: With the mask, learning is slower, but the cumulative reward plot is also pretty good

  Args:
    anchor (jax.Array): (..., D)
    positive (jax.Array): (..., D)
    negative (jax.Array): (..., D)
    negative_mask (jax.Array): (...)
    margin (float): Defaults to 0.5

  Returns:
    jax.Array: (B, T)
  """
  anchor = anchor / jnp.linalg.norm(anchor, axis=-1, keepdims=True).clip(eps) # (..., D)
  positive = positive / jnp.linalg.norm(positive, axis=-1, keepdims=True).clip(eps) # (..., D)
  negative = negative / jnp.linalg.norm(negative, axis=-1, keepdims=True).clip(eps) # (..., D)
  sim_anc_pos = jnp.sum(anchor * positive, axis=-1)  # (...)
  sim_anc_neg = jnp.sum(anchor * negative, axis=-1)  # (...)
  # loss = jnp.maximum(eps, sg(negative_mask) * (sim_anc_neg - sim_anc_pos + margin)) # (...)
  loss = jnp.maximum(0, sg(negative_mask) * sim_anc_neg - sim_anc_pos + margin) # (...)
  # jax.lax.cond(jnp.isfinite(loss).all(), lambda _: loss, lambda _: jax.debug.print("Found NaN or Inf"))
  # loss = jnp.maximum(0, sim_anc_neg - sim_anc_pos + margin)
  # print(f"[tripletloss] is finite: {jnp.all(jnp.isfinite(loss))}")
  # return jnp.zeros(()) # (...)
  return loss # (...)


def min_max_norm(arr: jax.Array | np.ndarray, axis=-1, eps=1e-5):
  min_arr = arr.min(axis=axis, keepdims=True)
  max_arr = arr.max(axis=axis, keepdims=True)
  out = (arr - min_arr) / (max_arr - min_arr).clip(eps)
  return out


def batched_1D_interpolation(vector: jax.Array, dim: int) -> jax.Array:
  """Perform 1D interpolation of 1D vector

  Args:
      vector (jax.Array): (batch, original_dim)
      dim (int): the target dimension

  Returns:
      jax.Array: (batch, dim)
  """
  B, S = vector.shape # (batch, source)
  T = dim # target
  # source_idx = jnp.linspace(0, S - 1, num=S)
  target_idx = jnp.linspace(0, S - 1, num=T)
  left_idx = jnp.floor(target_idx).astype(jnp.int32)
  right_idx = jnp.clip(left_idx + 1, 0, S - 1)
  weights = target_idx - left_idx
  left_values = vector[:, left_idx]
  right_values = vector[:, right_idx]
  interpolated = (1 - weights) * left_values + weights * right_values
  return interpolated


def masked_fill_other(x: jax.Array, mask: jax.Array, other=0) -> jax.Array:
  """Return an output with masked condition, with non-masked value
    be the other value

  Args:
      x (jax.Array): _description_
      mask (jax.Array): _description_
      other (int, optional): _description_. Defaults to 0.

  Returns:
      jax.Array: _description_
  """
  return jnp.where(mask, x, jnp.broadcast_to(other, x.shape))


def masked_fill(x: jax.Array, mask: jax.Array, value=0) -> jax.Array:
  """Return an output with masked condition, with non-masked value
    be the other value

  Args:
      x (jax.Array): _description_
      mask (jax.Array): _description_
      other (int, optional): _description_. Defaults to 0.

  Returns:
      jax.Array: _description_
  """
  return jnp.where(mask, jnp.broadcast_to(value, x.shape), x)


def resize_2d(x: jax.Array, size: Tuple[int, int], method: str = 'bilinear'):
  *B, H, W, C = x.shape
  return jax.image.resize(x, (*B, *size, C), method=method)


def reflection_pad_2d(x: jax.Array, pad: int):
  *B, H, W, C = x.shape
  pad_width = [(0, 0) for _ in range(len(B))] + [(pad, pad), (pad, pad), (0, 0)]
  return jnp.pad(x, pad_width, mode='reflect') # equals to reflection pad 2D


def pad_2d(x: jax.Array, pad: int):
  *B, H, W, C = x.shape
  pad_width = [(0, 0) for _ in range(len(B))] + [(pad, pad), (pad, pad), (0, 0)]
  return jnp.pad(x, pad_width, mode='constant', constant_values=0) # equals to reflection pad 2D


def bce(inputs: jax.Array, target: jax.Array, eps=1e-8) -> jax.Array:
  """Binary cross entropy loss with inputs as probabilities

  Args:
      inputs (jax.Array): _description_
      target (jax.Array): _description_
      eps (_type_, optional): _description_. Defaults to 1e-8.

  Returns:
      jax.Array: _description_
  """
  return - (target * (jnp.log(inputs.clip(eps))) + (1 - target) * jnp.log((1 - inputs).clip(eps)))


def bce_with_logits(logits: jax.Array, target: jax.Array, eps=1e-8) -> jax.Array:
  """Binary cross entropy loss with inputs as logits

  Args:
      logits (jax.Array): (*B, dim)
      target (jax.Array): (*B, dim)
      eps (_type_, optional): _description_. Defaults to 1e-8.

  Returns:
      jax.Array: _description_
  """
  # return - (target * (jax.nn.log_sigmoid(logits)) + (1 - target) * (jax.nn.log_sigmoid(-logits)))
  return jax.nn.softplus(-logits) + (logits - jax.nn.softplus(logits)) * target # more stable version


def focal_loss_with_logits(logits: jax.Array, targets: jax.Array,
    gamma: float = 2.0, eps: float = 1e-8) -> jax.Array:
  """Compute Focal Loss from BCE.

  Args:
    logits (jax.Array): Logits before sigmoid activation. (...) or (..., classes)
    targets (jax.Array): Ground truth labels (0 or 1). (...) or (..., classes)
    gamma (float): Focusing parameter gamma (default: 2.0).
    eps (float): Small value for numerical stability.

  Returns:
    jax.Array: Focal loss for each sample.
  """
  probs = jax.nn.sigmoid(logits)  # Convert logits to probabilities
  probs = jnp.clip(probs, eps, 1 - eps)  # Prevent log(0) issues
  # Standard BCE Loss
  bce_loss = - (targets * jnp.log(probs) + (1 - targets) * jnp.log(1 - probs))
  # Focal Loss scaling factor
  focal_weight = (1 - probs) ** gamma * targets + probs ** gamma * (1 - targets)
  # Apply focal weighting
  return focal_weight * bce_loss


def focal_loss_multiclass_with_logits(
    logits: jax.Array,
    targets: jax.Array,
    gamma: float = 2.0,
    eps: float = 1e-8,
    class_weights: jax.Array | None = None  # shape: (num_classes,)
) -> jax.Array:
    """
    Multi-class Focal Loss with logits.
    This is the focal loss for multi-class classification tasks that have
      probabilities summed to 1.0

    Args:
      logits: (batch_size, num_classes) — raw model outputs
      targets: (batch_size,) — integer class labels (0 to num_classes-1)
      gamma: focusing parameter
      eps: small value to prevent log(0)
      class_weights: optional array of shape (num_classes,)

    Returns:
      loss: (batch_size,) — focal loss per sample
    """
    log_probs = jax.nn.log_softmax(logits)  # (B, C)
    probs = jnp.exp(log_probs)              # (B, C)

    # Gather the predicted prob and log-prob for the target class
    targets_one_hot = jax.nn.one_hot(targets, num_classes=logits.shape[-1])
    pt = jnp.sum(probs * targets_one_hot, axis=-1)           # (B,)
    pt = jnp.clip(pt, eps, 1.0)                              # prevent log(0)
    log_pt = jnp.sum(log_probs * targets_one_hot, axis=-1)   # (B,)

    # Apply class weights (gather per-sample)
    if class_weights is not None:
        weight_per_sample = class_weights[jnp.int32(targets)]     # (B,)
    else:
        weight_per_sample = 1.0

    loss = -weight_per_sample * ((1 - pt) ** gamma) * log_pt  # (B,)

    return loss  # You can .mean() outside if needed


def nll_loss(logits: jax.Array, labels: jax.Array) -> jax.Array:
  """Negative log likelihood loss

  Args:
      logits (jax.Array): (B, T, C)
      labels (jax.Array): (B, T, C)

  Returns:
      jax.Array: (B, T)
  """
  assert logits.shape == labels.shape, (logits.shape, labels.shape)
  # return -jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)
  return jnp.sum(labels * (jnp.log(labels + 1e-8) - jax.nn.log_softmax(logits)), axis=-1)


def symlog(x):
  return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x):
  return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def get_pixel_value(img: jax.Array, x: jax.Array, y: jax.Array):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.

    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)

    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    batch_size, height, width = x.shape
    batch_idx = jnp.arange(0, batch_size)
    batch_idx = batch_idx[..., None, None]
    b = jnp.tile(batch_idx, (1, height, width)) # (B, H, W)
    return img[b, y, x]


def l2_norm(x: jax.Array):
  dtype = x.dtype
  L = x.shape[-1]
  epsilon = 1e-6
  x = jnp.sqrt((x**2).sum(-1, keepdims=True) + epsilon).repeat(L, -1) # (*B, 1) -> (*B, L)
  return x.astype(dtype)


def gelu_tanh(x):
  # Constants used in the approximation
  sqrt_2_over_pi = jnp.sqrt(2 / jnp.pi)
  coeff = 0.044715
  # GELU approximation formula
  return 0.5 * x * (1 + jnp.tanh(sqrt_2_over_pi * (x + coeff * jnp.power(x, 3))))


def categorical_cross_entropy(logits, labels, onehot=True):
  # logits: (*B, E), label: (*B)
  log_probs = jax.nn.log_softmax(logits)
  if onehot:
    _labels = jax.nn.one_hot(labels, logits.shape[-1]) # (*B, E)
  else:
    _labels = labels # (*B, E)
  loss = -jnp.sum(log_probs * _labels, axis=-1) # (*B)
  return loss


def image_correlation(x1: jax.Array, x2: jax.Array):
  assert x1.shape == x2.shape, (x1.shape, x2.shape)
  B, H, W, C = x1.shape
  x1 = x1.transpose([0, 3, 2, 1]) # (B, C, W, H)
  x1 = x1.reshape([B, C, W*H]) # (B, C, WH)

  x2 = x2.reshape([B, C, H*W])
  x2 = x2.transpose([0, 2, 1]) # (B, HW, C)

  corr = jnp.einsum("bij,bjk->bik", x2, x1) # (B, HW, WH)
  corr = corr.reshape([B, H, W, W*H]) # (B, H, W, WH)
  return corr

def sequence_correlation(x1: jax.Array, x2: jax.Array):
  # Calculate the Pearson correlation coefficient
  assert x1.shape == x2.shape, (x1.shape, x2.shape)
  x1 = normalize(x1, axis=-1)
  x2 = normalize(x2, axis=-1)
  corr = jnp.einsum("bij,bkj->bik", x1, x2) # (B, T, T)
  return corr # similar to the computation of the attention matrix


def expand_like(xs: jax.Array, ys: jax.Array) -> jax.Array:
  """Expand xs to the shape of ys"""
  expanded = jnp.expand_dims(xs, list(range(xs.ndim, ys.ndim)))
  expanded = jnp.broadcast_to(expanded, ys.shape)  # broadcast to the shape of ys
  return expanded


def where(condition, xs, ys):
  """

  Args:
      condition (Tree): any tree with shape (*some_shape,)
      xs (Tree): any tree with shape (*some_shape, *dim)
      ys (Tree): any tree with shape (*some_shape, *dim)

  Returns:
      Tree: same shape as xs and ys. Resulted masked value with mask broadcasted to the shape of xs and ys
  """
  assert condition.dtype == bool, condition.dtype
  def fn(x, y):
    assert x.shape == y.shape, (x.shape, y.shape)
    expanded = jnp.expand_dims(condition, list(range(condition.ndim, x.ndim)))
    return jnp.where(expanded, x, y)
  return jax.tree.map(fn, xs, ys)


def sg_where(x: jax.Array, should_sg: jax.Array | bool) -> jax.Array:
  """Applies stop-gradient to x if cond is True."""
  if isinstance(should_sg, bool):
    return sg(x) if should_sg else x
  sgx = sg(x)
  return where(should_sg, sgx, x)


def mask(xs, mask):
  """Resulted masked value with mask broadcasted to the shape of xs
    negative mask values will become zeros.

  Args:
      xs (Tree): any tree with shape (*some_shape, *dim)
      mask (jax.Array): (*some_shape,)

  Returns:
      Tree: same shape as xs. Resulted masked value with mask broadcasted to the shape of xs
        negative mask values will become zeros.
  """
  return where(mask, xs, jax.tree.map(jnp.zeros_like, xs))


def dropout(x, prob, training):
  if not prob or not training:
    return x
  keep = jax.random.bernoulli(nj.seed(), 1.0 - prob, x.shape)
  return x * keep / (1.0 - prob)


def sinusoidal(d_model: int, shape: Tuple[int, ...]):
  """
  Computes sinusoidal positional embeddings in JAX.

  Args:
    d_model (int): The embedding dimension.
    shape (Tuple[int, ...]): The event shape, which multiplies to the sequence length.

  Returns:
    jax.Array: A tensor of shape (*shape, d_model) containing the positional encodings.
  """
  assert len(shape) > 0, "Shape must be non-empty"
  assert d_model % 2 == 0, f"Hidden dim must be divisible by 2, got d_model = {d_model}"
  T = math.prod(shape)  # Compute total sequence length
  position = jnp.arange(T)[:, None]  # Shape: (T, 1)
  div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-jnp.log(10000.0) / d_model))
  # Compute sin and cos values separately
  sin_values = jnp.sin(position * div_term)  # Shape: (T, d_model/2)
  cos_values = jnp.cos(position * div_term)  # Shape: (T, d_model/2)
  # Interleave sin and cos values correctly
  pe = jnp.zeros((T, d_model))
  pe = jnp.concatenate([sin_values, cos_values], axis=-1).reshape(T, d_model)
  return pe.reshape((*shape, d_model))  # Shape: (*shape, d_model)


def rms(xs):
  """Compute root mean square for the whole tree

  Args:
      xs (_type_): _description_

  Returns:
      _type_: _description_
  """
  xs = jax.tree.leaves(xs)
  count = sum(x.size for x in xs)
  sumsq = jnp.stack([f32(jnp.square(x).sum()) for x in xs]).sum()
  return jnp.sqrt(sumsq / f32(count))


def normalize(x, p=2, axis=-1, eps=1e-12):
  norm = jnp.linalg.norm(x, ord=p, axis=axis, keepdims=True)
  return x / (norm + eps)


def cosine_similarity(a: jax.Array, b: jax.Array) -> jax.Array:
  """ Compute batched cosine similarity between a and b

  Args:
    a (jax.Array): (B, D)
    b (jax.Array): (B, D)

  Returns:
    jax.Array: (B,)
  """
  return jnp.einsum("BX,BX->B", a, b) / (jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(b, axis=-1))


def all_pair_cosine_similarity(a: jax.Array, b: jax.Array) -> jax.Array:
  """ Compute batched all pair cosine similarity between a and b

  Args:
    a (jax.Array): (B, T1, D)
    b (jax.Array): (B, T2, D)

  Returns:
    jax.Array: (B, T1, T2)
  """
  norm_a = normalize(a, p=2, axis=-1)
  norm_b = normalize(b, p=2, axis=-1)
  return jnp.einsum("BXD,BYD->BXY", norm_a, norm_b)


def triplet_contrastive_loss(latent: jax.Array, pos: jax.Array, neg: jax.Array, margin=0.5) -> jax.Array:
  """Compute batched all pair contrastive loss

  Args:
    latent (jax.Array): (B, D)
    pos (jax.Array): (B, T, D)
    neg (jax.Array): (B, T, D)

  Returns:
    torch.Tensor: (B, T)
  """
  latent_norm = normalize(latent, p=2, axis=-1)
  positive_norm = normalize(pos, p=2, axis=-1)
  negative_norm = normalize(neg, p=2, axis=-1)
  pos_sim = jnp.einsum('BD,BND->BN', latent_norm, positive_norm)  # (batch, n_examples)
  neg_sim = jnp.einsum('BD,BND->BN', latent_norm, negative_norm)  # (batch, n_examples)
  # Compute triplet loss (we want pos_sim > neg_sim + margin)
  loss = jnp.clip(margin + neg_sim - pos_sim, min=0) # (batch, n_examples)
  return loss, pos_sim, neg_sim


def n_pair_with_margin_contrastive_loss(latent: jax.Array, pos: jax.Array, neg: jax.Array,
    margin : float = 1.0, logit_scale: Optional[jax.Array] = None) -> jax.Array:
  """compute n-pair contrastive loss

  Args:
      latent (jax.Array): (batch, dim)
      pos (jax.Array): (batch, dim)
      neg (jax.Array): (batch, n_examples, dim)
      margin (float, optional): _description_. Defaults to 1.0.

  Returns:
      jax.Array: loss (batch,), pos_sim (batch,), neg_sim (batch, n_examples)
  """
  latent_norm = normalize(latent, p=2, axis=-1)
  positive_norm = normalize(pos, p=2, axis=-1)
  negative_norm = normalize(neg, p=2, axis=-1)
  pos_sim = jnp.einsum('BD,BD->B', latent_norm, positive_norm)  # (batch,)
  pos_sim_single = pos_sim[:, None] # (B, 1)
  neg_sim = jnp.einsum('BD,BND->BN', latent_norm, negative_norm)  # (batch, n_examples)
  if logit_scale is not None:
    _logit_scale = logit_scale.exp()
    pos_sim_single *= _logit_scale
    neg_sim *= _logit_scale
  # Compute n_pair_loss
  loss = 1 + jnp.exp(margin + neg_sim - pos_sim_single).sum(-1) # (batch,)
  loss = jnp.log(loss.clip(min=1e-5))
  return loss, pos_sim, neg_sim


def weights_norm(weights: jax.Array, target_norm: float, axis: int = 1, order: int = 1, eps: float = 1e-8) -> jax.Array:
  """
  Adapted from https://github.com/NACLab/ngc-learn/blob/main/ngclearn/utils/model_utils.pys
  Normalizes the values in matrix to have a particular norm across each vector span.
  Check out https://github.com/NACLab/ngc-museum/blob/main/exhibits/olshausen_sc/sparse_coding.py

  Args:
      weights: (2D) weights matrix to normalize (M, N)

      target_norm: target norm for each row/column of data matrix. x: (B, dim) W (dim, out) -> y (B, out): normalize W in the out dimension

      order: order of norm to use in normalization (Default: 1);
          note that `ord=1` results in the L1-norm, `ord=2` results in the L2-norm

      axis: 0 (apply to column vectors), 1 (apply to row vectors)

  Returns:
      a normalized value matrix
  """
  assert weights.ndim == 2, f"Data must be 2D, got {weights.shape}"
  assert order in (1, 2), f"Order must be 1 or 2, got {order}"
  if order == 2:
    # denominator is L2 norm
    wOrdSum = jnp.sqrt((weights**2).sum(axis=axis, keepdims=True)) + eps # (M, 1) or (1, N)
  else:
    # denominator is L1 norm
    wOrdSum = jnp.abs(weights).sum(axis=axis, keepdims=True) + eps # (M, 1) or (1, N)
  return weights * (target_norm / wOrdSum) # (M, N) * (M, 1) or (1, N)

def huber_loss(pred: jax.Array, target: jax.Array, delta: float = 1.0) -> jax.Array:
  """Compute Huber loss between predictions and targets.

  Args:
      pred (jax.Array): Predictions. (...)
      target (jax.Array): Targets. (...)
      delta (float, optional): Threshold for switching between L1 and L2 loss. Defaults to 1.0.

  Returns:
      jax.Array: Huber loss. (...)
  """
  error = pred - target
  abs_error = jnp.abs(error)
  quadratic = jnp.minimum(abs_error, delta)
  linear = abs_error - quadratic
  return 0.5 * quadratic**2 + delta * linear