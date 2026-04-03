from typing import Union, Tuple, List

import numpy as np
import jax
import jax.numpy as jnp

def create_3d_causal_attention_mask(batch_size: int, sequence_length: int) -> jax.Array:
  """Make causal mask used for bi-directional self-attention. This masking is compatible with
    the attention mechanism in `nn.base:Attention`

  Args:
      batch_size (int): The batch size.
      sequence_length (int): The sequence length.

  Returns:
      jax.Array:  A tensor of shape (batch_size, target_len, target_len),
        where False corresponds to the unchanged value in the attention matrix
        while True represents the value to be replaced with negative infinity.
        This is a negative mask. Negative masks: False for valid positions,
        True for padding/value to be replaced with -inf
  """
  assert sequence_length > 0
  mask = jnp.tril(jnp.ones((sequence_length, sequence_length)), k=0)[None] == 1 # (1, T, T)
  mask = mask.repeat(batch_size, axis=0) # (B, T, T)
  return ~mask

# backward compatibility
def create_3d_attention_mask_from_2d_positive_mask(source_mask: jax.Array, target_length: int | None = None) -> jax.Array:
  return create_3d_attention_mask_from_2d_positive_mask_rows_columns(source_mask, target_length)


# NOTE: This only cross out the columns of the attention matrix, used for cross-attention
def create_3d_attention_mask_from_2d_positive_mask_columns(source_mask: jax.Array, target_length: int | None = None) -> jax.Array:
  """
  Convert a 2D positive mask to a 3D attention mask that is compatible with the attention mechanism
    in `nn.base:Attention`.
  [B, source_length] -> [B, target_length, source_length]
  Tl;DR: We always mask the key dimension, not the queries (mask the source, not the target -- mask the
    column of the attention matrix, not the row)

  Args:
      source_mask (jax.Array): A 2D tensor of shape (batch_size, source_len) with 1 for valid positions and 0 for padding.

  Returns:
      jax.Array:  A tensor of shape (batch_size, target_len, source_len),
        where False corresponds to the unchanged value in the attention matrix
        while True represents the value to be replaced with negative infinity.
        This is a negative mask. Negative masks: False for valid positions,
        True for padding/value to be replaced with -inf
  """
  assert source_mask.ndim == 2
  source_length = source_mask.shape[1]
  if target_length is None:
    target_length = source_length
  mask = jnp.astype(source_mask, jnp.bool_)
  mask = jnp.repeat(mask[:, None, :], target_length, axis=1) # (B, T, S)
  mask = ~mask
  mask = compute_fallback_mask(mask)
  return mask


# NOTE: This cross out the rows and columns of the attention matrix, used for self-attention
def create_3d_attention_mask_from_2d_positive_mask_rows_columns(source_mask: jax.Array) -> jax.Array:
  """
  Convert a 2D positive mask to a 3D attention mask that is compatible with the attention mechanism
    in `nn.base:Attention`.
  [B, source_length] -> [B, target_length, source_length]
  Tl;DR: We always mask the key dimension, not the queries (mask the source, not the target -- mask the
    column of the attention matrix, not the row)

  Args:
      source_mask (jax.Array): A 2D tensor of shape (batch_size, source_len) with 1 for valid positions and 0 for padding.

  Returns:
      jax.Array:  A tensor of shape (batch_size, target_len, source_len),
        where False corresponds to the unchanged value in the attention matrix
        while True represents the value to be replaced with negative infinity.
        This is a negative mask. Negative masks: False for valid positions,
        True for padding/value to be replaced with -inf
  """
  assert source_mask.ndim == 2
  # source_length = source_mask.shape[1]
  mask = jnp.astype(source_mask, jnp.bool_)
  mask = mask[:, None, :] & mask[:, :, None] # (B, 1, S) & (B, S, 1) -> (B, S, S)
  mask = ~mask
  mask = compute_fallback_mask(mask)
  return mask

def compute_fallback_mask(mask: jax.Array) -> jax.Array:
  """False is the valid position, True is the padding position and will be replaced with -inf

  Args:
      mask (jax.Array): (B, T, S)

  Returns:
      jax.Array: (B, T, S)
  """
  # compute fallback mask
  # Identify rows where all keys are masked (True)
  all_masked = jnp.all(mask, axis=-1, keepdims=True)  # shape: (B, T, 1)
  # Fallback mask: unmask position 0
  fallback = jnp.zeros_like(mask)
  fallback = fallback.at[:, :, 0].set(False)  # allow attention to position 0
  # Use jnp.where to apply fallback
  mask = jnp.where(all_masked, fallback, mask)
  return mask


def combine_masks(*masks: List[jax.Array]) -> jax.Array:
  """Combine a list of negative masks into a single mask.
  Negative masks: False for valid positions, True for padding/value to be replaced with -inf

  Args:
    masks (List[jax.Array]): A list of masks to combine.

  Returns:
    jax.Array: A combined mask.
  """
  return jnp.logical_or.reduce(jnp.stack(masks, axis=-1), axis=-1)
