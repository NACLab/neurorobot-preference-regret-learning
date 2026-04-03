
import jax
import jax.numpy as jnp

def take_patches_by_masks(x: jax.Array, masks: list[jax.Array]):
  """
  :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
  :param masks: list of tensors of shape [B, K] containing indices of K patches in [N] to keep
  """
  B, N, D = x.shape
  all_x = []
  for m in masks:
    _, K = m.shape
    mask_keep = m[..., None] # (B, K, 1)
    all_x.append(jnp.take_along_axis(x, mask_keep, axis=1))
  return all_x

