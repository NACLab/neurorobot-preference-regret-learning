"""
File: base.py
Author: Viet Nguyen
Date: 2025-03-18

Description: Implement the base class for all agents
"""

class BaseActiveAgent:

  def __init__(self, obs_space, act_space, config):
    pass

  def init_train(self, batch_size):
    raise NotImplementedError('init_train(batch_size) -> carry')

  def init_report(self, batch_size):
    raise NotImplementedError('init_report(batch_size) -> carry')

  def init_policy(self, batch_size):
    raise NotImplementedError('init_policy(batch_size) -> carry')

  def train(self, carry, data):
    raise NotImplementedError('train(carry, data) -> carry, out, metrics')

  def report(self, carry, data):
    raise NotImplementedError('report(carry, data) -> carry, metrics')

  def policy(self, carry, obs, mode):
    raise NotImplementedError('policy(carry, obs, mode) -> carry, act, out')

  def stream(self, st):
    raise NotImplementedError('stream(st) -> st')

  def save(self):
    raise NotImplementedError('save() -> data')

  def load(self, data):
    raise NotImplementedError('load(data) -> None')

