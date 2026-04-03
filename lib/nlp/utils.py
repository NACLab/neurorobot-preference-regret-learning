"""
File: utils.py
Author: Viet Nguyen
Date: 2024-10-24

Description:
"""

from typing import List
import numpy as np

def add_text(sequences: List[str], txt: str, mode="post") -> List[str]:
  if mode == 'post':
    return [seq + txt for seq in sequences]
  elif mode == 'pre':
    return [txt + seq for seq in sequences]
  else:
    raise ValueError(f"Mode should be `pre` or `post`. Got {mode}")

def pad_sequences(sequences, max_len=None, padding="post", pad_token=0):
  """
  Pad a list of tokenized sequences to the same length.

  Parameters:
  - sequences: List of lists, where each inner list is a sequence of tokens.
  - max_len: The length to pad the sequences to. If None, it defaults to the longest sequence length.
  - padding: "post" to add padding at the end, "pre" to add padding at the beginning.
  - pad_token: The token to use for padding.

  Returns:
  - A NumPy array with padded sequences.
  """
  # Find the length to pad to
  if max_len is None:
    max_len = max(len(seq) for seq in sequences)

  # Initialize a NumPy array with pad tokens
  padded_sequences = np.full((len(sequences), max_len), pad_token)

  for i, seq in enumerate(sequences):
    if padding == "post":
      padded_sequences[i, :len(seq)] = seq  # Pad at the end
    elif padding == "pre":
      padded_sequences[i, -len(seq):] = seq  # Pad at the beginning
    else:
      raise ValueError("Padding type must be 'post' or 'pre'")

  return padded_sequences

def decode_utf8(bytes_array: np.ndarray) -> str:
  """Given a 1D array of utf8 bytes, decode them into string
  """
  # text: (text_length,)
  """Utility to decode encoded language instruction"""
  return bytes(bytes_array[np.where(bytes_array != 0)].tolist()).decode("utf-8")

