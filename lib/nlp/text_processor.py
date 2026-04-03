# %%

from abc import ABC, abstractmethod
from typing import Optional, Sequence
from functools import partial as bind
import numpy as np
import functools

def build_text_processor(config):
  if 'text_processor' not in config or not config.text_processor:
    return None
  maxseqlen = config.max_sequence_length if 'max_sequence_length' in config  else None
  processor = {
    "hf": bind(HFProcessor, max_sequence_length=maxseqlen)
  }[config.text_processor](**config.text_processor_kwargs.get(config.text_processor, {}))

  return processor


def build_label_text_processor(config):

  processor = {
    "hf": bind(HFProcessor, max_sequence_length=config.max_label_sequence_length)
  }[config.text_processor](**config.text_processor_kwargs.get(config.text_processor, {}))

  return processor

class TextProcessor(ABC):
  """
  Base class for text tokenization or text embedding.
  """

  @property
  def vocab_size(self) -> int:
    s = self.__getattribute__("_vocab_size")
    if not s:
      raise NotImplementedError("Set this in sub class")
    return s

  @vocab_size.setter
  def vocab_size(self, value):
    self._vocab_size = value

  @property
  def max_sequence_length(self) -> int:
    s = self.__getattribute__("_max_sequence_length")
    if not s:
      raise NotImplementedError("Set this in sub class")
    return s

  @max_sequence_length.setter
  def max_sequence_length(self, value):
    self._max_sequence_length = value

  @abstractmethod
  def encode(self, strings: Sequence[str] | str) -> dict:
    raise NotImplementedError(f"Returns: dict(input_ids=..., attention_mask=...)")

  # @abstractmethod
  def decode(self):
    raise NotImplementedError(f"...")


class DefaultTextProcessor(TextProcessor):

  def __init__(self, max_sequence_length: int = 256):
    # encode characters to number
    self.max_sequence_length = max_sequence_length
    self.vocab_size = 256 # 256 is the number of characters in the ASCII table

  def encode(self, strings: Sequence[str] | str) -> dict:
    if isinstance(strings, str):
      strings = [strings]
    encoded = [(list(map(ord, string)) + [0] * self.max_sequence_length)[:self.max_sequence_length] for string in strings]
    encoded = np.asarray(encoded, dtype=np.int32)
    return {
      'input_ids': encoded,
      'attention_mask': np.array(encoded != 0, dtype=bool)
    }

  def decode(self, input_ids: np.ndarray) -> str:
    raise NotImplementedError("Not implemented")


class HFProcessor(TextProcessor):
  def __init__(self, tokenizer_name: str, max_sequence_length: Optional[int] = None, padding: Optional[str] = None,
      truncation: bool = True, return_tensors: str = "np", **kwargs):
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    config = AutoConfig.from_pretrained(tokenizer_name)
    # Optional[dict] = {
    #   "max_length": 64,
    #   "padding": "max_length",
    #   "truncation": True,
    #   "return_tensors": "np",
    # }
    self.kwargs = {
      "truncation": truncation if max_sequence_length is not None else False,
      "return_tensors": return_tensors,
      **kwargs
    }
    max_sequence_length is not None and self.kwargs.update({"max_length": max_sequence_length})
    padding is not None and self.kwargs.update({"padding": padding})
    # self.vocab_size = self.tokenizer.vocab_size
    self.vocab_size = config.vocab_size
    self.max_sequence_length = max_sequence_length

  def encode(self, strings: Sequence[str] | str):
    if isinstance(strings, str):
      strings = [strings]
    # this creates another nested layer with "input_ids", "attention_mask", "pixel_values" etc.
    out = self.tokenizer(strings, **self.kwargs)
    return {
      'input_ids': np.asarray(out['input_ids'], dtype=np.int32),
      'attention_mask': np.asarray(out['attention_mask'], dtype=bool)
    }

  def decode(self, input_ids: np.ndarray) -> str:
    if isinstance(input_ids, np.ndarray):
      input_ids = input_ids.tolist()
    # Remove batch dimension if present
    if isinstance(input_ids, list) and len(input_ids) > 0 and isinstance(input_ids[0], list):
      input_ids = input_ids[0]
    return self.tokenizer.decode(input_ids, skip_special_tokens=True)


if __name__ == '__main__':
  from core import Config
  c = Config({"text_processor": "hf", "max_sequence_length": 256, "text_processor_kwargs": {"hf": {"tokenizer_name": "google/flan-t5-small", "padding": "max_length"}}})
  print(c)
  proc = build_text_processor(c)
  out = proc.encode(["hello"]) # Similar to out = proc.encode("hello")
  print(out.keys())
  print(out)
  print(proc.vocab_size)
