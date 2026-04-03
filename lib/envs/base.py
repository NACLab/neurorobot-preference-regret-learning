"""
File: base.py
Author: Viet Nguyen
Date: 2025-03-12

Description: Base class for all environments
"""

from ..nlp.text_processor import TextProcessor
from typing import Dict, List
import numpy as np

class Env:

  def __repr__(self):
    return (
        f'{self.__class__.__name__}('
        f'obs_space={self.obs_space}, '
        f'act_space={self.act_space})')

  @property
  def obs_space(self):
    # The observation space must contain the keys is_first, is_last, and
    # is_terminal. Commonly, it also contains the keys reward and image. By
    # convention, keys starting with 'log/' are not consumed by the agent.
    raise NotImplementedError('Returns: dict of spaces')

  @property
  def act_space(self):
    # The action space must contain the reset key as well as any actions.
    raise NotImplementedError('Returns: dict of spaces')

  def step(self, action):
    raise NotImplementedError('Returns: dict')

  def close(self):
    pass

  def render(self) -> Dict[str, np.ndarray] | np.ndarray | List[np.ndarray]:
    raise NotImplementedError('Returns: dict of images or image or list of images')


class ActiveEnv(Env):
  """Same as Env, just name it differently"""
  pass

class InactiveEnv:

  def __repr__(self):
    return (
        f'{self.__class__.__name__}('
        f'obs_space={self.obs_space}'
        f', label_space={self.label_space})')

  @property
  def obs_space(self):
    # The observation space must contain the keys is_first (first in the sequence),
    #   is_last (last in the sequence), is_dataset_first (normally, we don't need this),
    #   is_dataset_last (end of the whole dataset).
    #   By convention, keys starting with 'log/' are not consumed by the agent.
    raise NotImplementedError('Returns: dict of spaces')

  @property
  def label_space(self):
    # The label space contain the output space of the model or the data field that we are predicting
    raise NotImplementedError('Returns: dict of spaces')

  def step(self):
    raise NotImplementedError('Returns: dict. Return data that combine both obs_space and label_space')

  def close(self):
    pass

  def render(self) -> Dict[str, np.ndarray] | np.ndarray | List[np.ndarray]:
    raise NotImplementedError('Returns: dict of images or image or list of images')


class ExpertDataset:

  def __init__(self, task: str, text_processor: TextProcessor, seed: int | None, **kwargs):
    pass

  @property
  def env(self) -> Env:
    raise NotImplementedError('Returns: Env')

  # def collect(self, final_env: Env, n_episodes: int = 1):
  #   raise NotImplementedError('Returns: list of list of dicts (episodes -> transitions -> step)')

  def dataset(self, final_env: Env):
    raise NotImplementedError('Yields: one transition')


class Wrapper:

  def __init__(self, env):
    self.env = env

  def __len__(self):
    return len(self.env)

  def __bool__(self):
    return bool(self.env)

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      # print(f'calling {name} on {self.env}')
      return getattr(self.env, name)
    except AttributeError:
      raise ValueError(name)

