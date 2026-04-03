
from typing import Dict, Callable, Any
import jax
import jax.numpy as jnp
import numpy as np
import collections
from functools import partial as bind

from lib import nn
# from lib.agent.active_simple import JAXAgent
from lib.agent.active import JAXAgent
from lib.common import Space


# NOTE: Working
class DummyAgent(JAXAgent):

  def __init__(self, obs_space, act_space, config, *, vocab_size: int, max_sequence_length: int):
    self.obs_space = obs_space
    self.act_space = act_space
    self.linear = nn.Linear(1, name='linear')
    self.opt = nn.Optimizer([self.linear], name='optimizer')

  @property
  def policy_keys(self):
    # return '^(enc|dyn|dec|pol)/'
    return '.*'

  @property
  def ext_space(self):
    spaces = {}
    # spaces['consec'] = Space(np.int32)
    spaces['stepid'] = Space(np.uint8, 20)
    spaces['is_expert'] = Space(bool)
    spaces['can_self_imitate'] = Space(bool)
    spaces['is_positive_memory'] = Space(bool)
    spaces['pref_label'] = Space(np.float32)
    return spaces

  def init_policy(self, batch_size):
    return ()

  def init_train(self, batch_size):
    return ()

  def init_report(self, batch_size):
    return ()

  def policy(self, carry, obs, mode='train'):
    batch_size = len(obs['is_first'])
    act = {
        k: jnp.stack([v.sample() for _ in range(batch_size)])
        for k, v in self.act_space.items() if k != 'reset'}
    return carry, act, {}

  def _loss(self, carry, data: Dict[str, jax.Array]):
    loss = ((self.linear(nn.cast(jnp.zeros((4, 2)))).astype(jnp.float32) - 1)**2).mean()
    return loss, {}

  def train(self, carry, data):
    opt_mets, mets = self.opt(self._loss, carry, data, has_aux=True)
    mets.update(opt_mets)
    return carry, {}, mets

  def report(self, carry, data):
    return carry, {}


