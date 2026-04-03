from ..common import print
from .base import ExpertDataset, Env
from .dummy import Dummy as DummyEnv
from ..nlp.text_processor import TextProcessor
import numpy as np
import h5py


class CommonExpertDataset(ExpertDataset):

  def __init__(self, task: str, text_processor: TextProcessor | None = None, seed: int | None = None, *,
      dataset_path: str | None = None, **env_kwargs):
    # Loading a dataset from a h5py file containing {episode_000000, episode_000001, ...}
    # Each episode is a dictionary with keys: {image_0, image_1, state, action, reward, ...}
    self._env = DummyEnv(None, size=(64, 64), length=100)
    print(f"[run] [expert] dataset_path: {dataset_path}")
    self._data = h5py.File(dataset_path, 'r')
    self._num_demos = len(self._data.keys())
    self._episode_keys = sorted(self._data.keys())
    assert self._num_demos > 0, f"No episodes found in {dataset_path}. Is your path correct?"

  @property
  def env(self) -> Env:
    return self._env

  def dataset(self, final_env: Env):
    i = 0
    while True:
      id = i % self._num_demos
      print(f"[run] [expert] fill_count (ep): {id} / {self._num_demos}", end='\r', color='green')
      episode_key = self._episode_keys[id]
      somekey = list(self._data[episode_key].keys())[0]
      episode_length = len(self._data[episode_key][somekey])
      episode_data = {k: np.asarray(v[()]) for k, v in self._data[episode_key].items()}
      for step in range(episode_length):
        trn = {k: np.asarray(v[step]) for k, v in episode_data.items()}
        if trn['is_last'] or trn['is_first']:
          if 'is_success' in trn and 'has_succeeded' in trn:
            trn['is_success'] = True
            trn['has_succeeded'] = True
        yield trn
      i = (i + 1) % self._num_demos

