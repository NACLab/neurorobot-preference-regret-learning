from typing import Dict, List
import numpy as np
from functools import partial as bind
from lib.common import streams

class EpisodeAggregator:
  def __init__(self):
    self.data: List[Dict[str, np.ndarray]] = []
    self.keys: List[str] = []

  def add(self, trans: Dict[str, np.ndarray]):
    self.data.append(trans)
    if len(self.data) == 1:
      self.keys = list(self.data[0].keys())

  def result(self, reset=True):
    out = {k: np.stack([d[k] for d in self.data]) for k in self.keys}
    if reset:
      self.reset()
    return out

  def reset(self):
    self.data = []
    self.keys = []


