# We can get the idea of this driver to be used in the actual simulation loop, we don't need to use the driver class
# NOTE: This was not used

import time

import cloudpickle
import numpy as np
import portal
import collections

from lib import tree

SPECIAL_FIELD_MESSAGE = "_special_field_message"
SPECIAL_FUNCTION_CALL_MESSAGE = "_special_function_call_message"

class Driver:

  def __init__(self, make_env_fns: list[callable], parallel=True, **kwargs):
    assert len(make_env_fns) >= 1
    self.parallel = parallel
    self.kwargs = kwargs
    self.length = len(make_env_fns)
    if parallel:
      import multiprocessing as mp
      context = mp.get_context()
      self.pipes, pipes = zip(*[context.Pipe() for _ in range(self.length)])
      self.stop = context.Event()
      fns = [cloudpickle.dumps(fn) for fn in make_env_fns]
      self.procs = [
          portal.Process(self._env_server, self.stop, i, pipe, fn, start=True)
          for i, (fn, pipe) in enumerate(zip(fns, pipes))]
      self.pipes[0].send(('act_space',))
      self.act_space = self._receive(self.pipes[0])
    else:
      self.envs = [fn() for fn in make_env_fns]
      self.act_space = self.envs[0].act_space
    self._on_steps = []
    self._on_episodes = []
    self.acts = None
    self.carry = None
    self.reset()

  def reset(self, init_policy=None):
    self.acts = {
        k: np.zeros((self.length,) + v.shape, v.dtype)
        for k, v in self.act_space.items()}
    self.acts['reset'] = np.ones(self.length, bool)
    self.carry = init_policy and init_policy(self.length)
    self.episode_data = [collections.defaultdict(list) for _ in range(self.length)]

  def close(self):
    if self.parallel:
      [proc.kill() for proc in self.procs]
    else:
      [env.close() for env in self.envs]

  def on_step(self, callback, id=None):
    if id is None:
      self._on_steps.append(callback)
    else:
      self._on_steps.insert(id, callback)

  def on_episode(self, callback, id=None):
    if id is None:
      self._on_episodes.append(callback)
    else:
      self._on_episodes.insert(id, callback)

  def __call__(self, policy, steps=0, episodes=0, modifier: callable = lambda x1: tree.map(lambda x2: x2, x1)):
    step, episode = 0, 0
    while step < steps or episode < episodes:
      step, episode = self._step(policy, step, episode, modifier)

  def _step(self, policy: callable, step, episode, modifier: callable):
    """
    This function is the core of the driver, it is called in the simulation loop
    to step the environment and call the policy.

    Args:
        policy (callable): _description_
        step (_type_): _description_
        episode (_type_): _description_
        modifier (callable | None, optional): callable to modify the transition ditionary for additional information
          to be fed into the (e.g. is_expert, can_self_imitate). Defaults to None.

    Returns:
        _type_: _description_
    """
    acts = self.acts
    assert all(len(x) == self.length for x in acts.values())
    assert all(isinstance(v, np.ndarray) for v in acts.values())
    acts = [{k: v[i] for k, v in acts.items()} for i in range(self.length)]
    if self.parallel:
      [pipe.send(('step', act)) for pipe, act in zip(self.pipes, acts)]
      obs = [self._receive(pipe) for pipe in self.pipes]
    else:
      obs = [env.step(act) for env, act in zip(self.envs, acts)]
    obs = {k: np.stack([x[k] for x in obs]) for k in obs[0].keys()}
    logs = {k: v for k, v in obs.items() if k.startswith('log/')}
    obs = {k: v for k, v in obs.items() if not k.startswith('log/')}
    assert all(len(x) == self.length for x in obs.values()), obs
    self.carry, acts, outs = policy(self.carry, obs, **self.kwargs)
    assert all(k not in acts for k in outs), (
        list(outs.keys()), list(acts.keys()))
    if obs['is_last'].any():
      mask = ~obs['is_last']
      acts = {k: self._mask(v, mask) for k, v in acts.items()}
    self.acts = {**acts, 'reset': obs['is_last'].copy()}
    trans = {**obs, **acts, **outs, **logs}
    trans = modifier(trans)
    if obs['is_first'].any():
      for i, first in enumerate(obs['is_first']):
        if first:
          self.episode_data[i].clear()
    for i in range(self.length):
      trn = tree.map(lambda x: x[i], trans)
      [self.episode_data[i][k].append(v) for k, v in trn.items()]
      [fn(trn, i, **self.kwargs) for fn in self._on_steps]
    # Call on_episode callbacks for episodes that have ended
    if obs['is_last'].any():
      for i, done in enumerate(obs['is_last']):
        if done:
          ep = {k: np.stack(v) for k, v in self.episode_data[i].items()}
          for fn in self._on_episodes:
            ep = fn(ep.copy(), i, **self.kwargs)
    step += len(obs['is_first'])
    episode += obs['is_last'].sum()
    return step, episode

  def get_field(self, var_name: str):
    if self.parallel:
      [pipe.send((f'{SPECIAL_FIELD_MESSAGE}_{var_name}',)) for pipe in self.pipes]
      return [self._receive(pipe) for pipe in self.pipes]
    else:
      try:
        result = [getattr(env, var_name) for env in self.envs]
      except:
        result = [None for _ in range(self.length)]
      return result

  def execute_function(self, function_name: str, *args, **kwargs):
    if self.parallel:
      [pipe.send((f'{SPECIAL_FUNCTION_CALL_MESSAGE}_{function_name}', args, kwargs)) for pipe in self.pipes]
      return [self._receive(pipe) for pipe in self.pipes]
    else:
      try:
        result = [getattr(env, function_name)(*args, **kwargs) for env in self.envs]
      except:
        result = [None for _ in range(self.length)]
      return result

  def _mask(self, value, mask):
    while mask.ndim < value.ndim:
      mask = mask[..., None]
    return value * mask.astype(value.dtype)

  def _receive(self, pipe):
    try:
      msg, arg = pipe.recv()
      if msg == 'error':
        raise RuntimeError(arg)
      assert msg == 'result'
      return arg
    except Exception:
      print('Terminating workers due to an exception.')
      [proc.kill() for proc in self.procs]
      raise

  @staticmethod
  def _env_server(stop, envid, pipe, ctor):
    try:
      ctor = cloudpickle.loads(ctor)
      env = ctor()
      while not stop.is_set():
        if not pipe.poll(0.1):
          time.sleep(0.1)
          continue
        try:
          msg, *args = pipe.recv()
        except EOFError:
          return
        if msg == 'step':
          assert len(args) == 1
          act = args[0]
          obs = env.step(act)
          pipe.send(('result', obs))
        elif msg == 'obs_space':
          assert len(args) == 0
          pipe.send(('result', env.obs_space))
        elif msg == 'act_space':
          assert len(args) == 0
          pipe.send(('result', env.act_space))
        elif msg.startswith(SPECIAL_FIELD_MESSAGE):
          assert len(args) == 0
          var_name = msg[len(SPECIAL_FIELD_MESSAGE)+1:]
          try:
            result = getattr(env, var_name)
          except Exception as e:
            result = None
          pipe.send(('result', result))
        elif msg.startswith(SPECIAL_FUNCTION_CALL_MESSAGE):
          assert len(args) == 2
          function_name, _args, _kwargs = args
          try:
            result = getattr(env, function_name)(*_args, **_kwargs)
          except Exception as e:
            result = None
          pipe.send(('result', result))
        else:
          raise ValueError(f'Invalid message {msg}')
    except ConnectionResetError:
      print('Connection to driver lost')
    except Exception as e:
      pipe.send(('error', e))
      raise
    finally:
      try:
        env.close()
      except Exception:
        pass
      pipe.close()
