"""
File: gym.py
Author: Viet Nguyen
Description: Gym environment wrapper, adapted from Danijar Hafner's gymnasium wrapper
"""

import functools

import gymnasium as gym
import numpy as np
import cv2

from ...common import *
from ..base import Env

class GymEnv(Env):

  def __init__(self, env, obs_key='state', act_key='action',
      img_obs_key='image', obs_render: bool = False, image_size: int = 64, **kwargs):
    # if obs_render is True, then this environment has an additional observation key 'image'
    # which is the rendered image of the environment
    # the image is a numpy array of shape (image_size, image_size, 3)

    if isinstance(env, str):
      self._env = gym.make(env, render_mode='rgb_array', **kwargs)
    else:
      assert not kwargs, kwargs
      self._env = env
    if obs_render:
      assert img_obs_key != obs_key, "img_obs_key and obs_key cannot be the same"
    self._img_obs_key = img_obs_key
    self._obs_render = obs_render
    self._image_size = image_size
    self._obs_dict = hasattr(self._env.observation_space, 'spaces')
    self._act_dict = hasattr(self._env.action_space, 'spaces')
    self._obs_key = obs_key
    self._act_key = act_key
    self._done = True
    self._info = None

  @property
  def env(self):
    return self._env

  @property
  def info(self):
    return self._info

  @functools.cached_property
  def obs_space(self):
    if self._obs_dict:
      spaces = self._flatten(self._env.observation_space.spaces)
    else:
      spaces = {self._obs_key: self._env.observation_space}
    if self._obs_render:
      spaces[self._img_obs_key] = Space(np.uint8, (self._image_size, self._image_size, 3), 0, 255)
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    return {
        **spaces,
        'reward': Space(np.float32),
        'is_first': Space(bool),
        'is_last': Space(bool),
        'is_terminal': Space(bool),
    }

  @functools.cached_property
  def act_space(self):
    if self._act_dict:
      spaces = self._flatten(self._env.action_space.spaces)
    else:
      spaces = {self._act_key: self._env.action_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    spaces['reset'] = Space(bool)
    return spaces

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      obs, self._info = self._env.reset()
      return self._obs(obs, 0.0, is_first=True)
    if self._act_dict:
      action = self._unflatten(action)
    else:
      action = action[self._act_key]
    obs, reward, terminated, truncated, self._info = self._env.step(action)
    self._done = terminated or truncated
    return self._obs(
        obs, reward,
        is_last=bool(self._done),
        is_terminal=bool(self._info.get('is_terminal', self._done)))

  def _obs(
      self, obs, reward, is_first=False, is_last=False, is_terminal=False):
    if not self._obs_dict:
      obs = {self._obs_key: obs}
    obs = self._flatten(obs)
    obs = {k: np.asarray(v) for k, v in obs.items()}
    if self._obs_render:
      img = self._env.render()
      img = cv2.resize(img, (self._image_size, self._image_size))
      obs[self._img_obs_key] = np.asarray(img)
    obs.update(
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal)
    return obs

  def render(self):
    image = self._env.render()
    assert image is not None
    return image

  def close(self):
    try:
      self._env.close()
    except Exception:
      pass

  # def _flatten(self, nest, prefix=None):
  #   result = {}
  #   for key, value in nest.items():
  #     # replace space with _S_
  #     key = key.replace(' ', '_S_')
  #     # key = prefix + '/' + key if prefix else key
  #     key = prefix + '__' + key if prefix else key
  #     if isinstance(value, gym.spaces.Dict):
  #       value = value.spaces
  #     if isinstance(value, dict):
  #       result.update(self._flatten(value, key))
  #     else:
  #       result[key] = value
  #   return result

  # def _unflatten(self, flat):
  #   result = {}
  #   for key, value in flat.items():
  #     # parts = key.split('/')
  #     parts = key.split('__') # we use '__' instead of '/' to avoid the issue of '/' in ninjax parsing
  #     node = result
  #     for part in parts[:-1]:
  #       if part not in node:
  #         # replace _S_ withspace
  #         part = part.replace('_S_', ' ')
  #         node[part] = {}
  #       node = node[part]
  #     parts[-1] = parts[-1].replace('_S_', ' ')
  #     node[parts[-1]] = value
  #   return result

  def _flatten(self, nest, prefix=None):
    result = {}
    for key, value in nest.items():
      key = prefix + '/' + key if prefix else key
      if isinstance(value, gym.spaces.Dict):
        value = value.spaces
      if isinstance(value, dict):
        result.update(self._flatten(value, key))
      else:
        result[key] = value
    return result

  def _unflatten(self, flat):
    result = {}
    for key, value in flat.items():
      parts = key.split('/')
      node = result
      for part in parts[:-1]:
        if part not in node:
          node[part] = {}
        node = node[part]
      node[parts[-1]] = value
    return result

  def _convert(self, space):
    if hasattr(space, 'n'):
      return Space(np.int32, (), 0, space.n - 1) # NOTE: low is inclusive, and high is inclusive, n is right-end exclusive
    elif hasattr(space, 'nvec'):
      return Space(np.int32, (len(space.nvec),), 0, space.nvec - 1) # NOTE: low is inclusive, and high is inclusive, nvec is right-end exclusive
    # return Space(space.dtype, space.shape, space.low, space.high)
    return Space(space.dtype, space.shape)
