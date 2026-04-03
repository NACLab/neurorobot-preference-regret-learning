"""
File: eval.py
Author: Viet Nguyen
Date: 2025-03-18

Description: Main run file for the evaluation of any agent in this project
"""

# %%

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from typing import Dict, Callable, Any, List
import jax
import jax.numpy as jnp
import numpy as np
import collections
import matplotlib.pyplot as plt
from io import BytesIO
import imageio
import re
import functools
from functools import partial as bind

from lib.nn import utils
from lib import Config
from lib import nn
from lib.envs.active_sensing import ActiveSensingEnv
from lib.common.config import load_config_from_yaml
from lib.common.logger import Logger, build_logger
from lib.common import when
from lib.utils import check_vscode_interactive
from lib.envs.build import build_env, build_expert_dataset, wrap_env
from lib.replay import build_replay
from lib.replay.replay import Replay, build_stream
from lib import streams
from lib.common.checkpoint import Checkpoint
from lib import common
from lib.common import print
from lib.nlp.text_processor import TextProcessor, build_text_processor
from lib.common.counter import Counter

from nprl.agents.build import build_agent
from nprl import callbacks
from nprl.utils import EpisodeAggregator
from nprl.driver import Driver

def run(config: Config):

  logger: Logger = build_logger(config)
  logdir = pathlib.Path(config.logdir)
  step = logger.step
  eval_step = Counter()
  logger.step = eval_step
  usage = common.Usage(**config.run.usage)
  epstats = common.Agg()
  episodes = collections.defaultdict(common.Agg)
  policy_fps = common.FPS()
  should_log = when.Clock(config.run.log_every)

  @common.timer.section('logfn')
  def logfn(tran, worker):
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add(f'score/{worker}', tran['reward'], agg='sum')
    episode.add(f'length/{worker}', 1, agg='sum')
    episode.add(f'rewards/{worker}', tran['reward'], agg='stack')
    episode.add(f'success_rate/{worker}', tran['success_rate'], agg='last')
    # if 'relative_stability' in tran:
    episode.add(f'relative_stability/{worker}', tran['relative_stability'], agg='last')
    # task_order = driver.get_field('tasks')[worker]
    # episode.add(f'task_order/{worker}', " -> ".join(task_order) if task_order is not None else "N/A", agg='last')
    task_text = driver.get_field('current_task_text')[worker]
    episode.add(f'task_text/{worker}', task_text if task_text is not None else "N/A", agg='last')
    for key, value in tran.items():
      value = np.asarray(value)
      if value.dtype == np.uint8 and value.ndim == 3:
        if worker == 0:
          episode.add(f'policy_{key}', value, agg='stack')
      elif key.startswith('log/'):
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key + f'/avg/{worker}', value, agg='avg')
        episode.add(key + f'/max/{worker}', value, agg='max')
        episode.add(key + f'/sum/{worker}', value, agg='sum')

    # log video render
    if config.log_video_render and (worker == 0 or worker == 1): # log the first two workers
      # log video render
      frame: np.ndarray | Dict[str, np.ndarray] = driver.execute_function('render')[worker]
      if frame is not None and not isinstance(frame, str):
        if isinstance(frame, Dict):
          for k, v in frame.items():
            episode.add(f'policy_render/{worker}/{k}', v, agg='stack')
        else:
          episode.add(f'policy_render/{worker}', frame, agg='stack')

    if tran['is_last']:
      # print("[logfn] episode end", color='yellow')
      result = episode.result()
      logger.add({
          f'score/{worker}': result.pop(f'score/{worker}'),
          f'length/{worker}': result.pop(f'length/{worker}'),
          f'success_rate/{worker}': result.pop(f'success_rate/{worker}'),
          f'relative_stability/{worker}': result.pop(f'relative_stability/{worker}'),
          # f'task_order/{worker}': result.pop(f'task_order/{worker}'),
          f'task_text/{worker}': result.pop(f'task_text/{worker}'),
      }, prefix='eval_episode')
      rew = result.pop(f'rewards/{worker}')
      if len(rew) > 1:
        result[f'reward_rate/{worker}'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      epstats.add(result)

  if config.text_processor:
    text_processor: TextProcessor = build_text_processor(config)
  else:
    text_processor = None

  agent = build_agent(config, text_processor)

  # Checkpoint processing
  checkpoint = Checkpoint(logdir / 'checkpoint.ckpt')
  checkpoint.step = step
  checkpoint.agent = agent
  if config.run.from_checkpoint:
    checkpoint.load(config.run.from_checkpoint)
  else:
    ckpt_path = logdir / 'checkpoint.ckpt'
    assert ckpt_path.exists(), f"Checkpoint file {ckpt_path} does not exist"
    checkpoint.load(ckpt_path)

  #### Driver Setup and Simulation Loop
  env_fns = [bind(build_env, config, i, text_processor=text_processor,
    multi_task=config.multi_task) for i in range(config.n_envs)]
  driver = Driver(env_fns, parallel=not config.run.debug)
  driver.on_step(lambda tran, _: eval_step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(logfn)

  # Start evaluation loop
  print('Start evaluation loop', color='green')

  # Initialize the driver with the policy
  policy = lambda *args: agent.policy(*args, mode='eval')

  # Simulation loop
  driver.reset(agent.init_policy)
  while eval_step < config.run.eval_steps:
    driver(policy, steps=10)

    print(f"Eval step: {int(eval_step)}", color='green', end='\r')

    if should_log(eval_step):
      logger.add(epstats.result(), prefix='eval_epstats')
      logger.add(usage.stats(), prefix='eval_usage')
      logger.add({'eval_fps/policy': policy_fps.result()})
      logger.add({'eval_timer': common.timer.stats()['summary']})
      logger.write()

  # Logging one last time
  logger.add(epstats.result(), prefix='eval_epstats')
  logger.add(usage.stats(), prefix='eval_usage')
  logger.add({'eval_fps/policy': policy_fps.result()})
  logger.add({'eval_timer': common.timer.stats()['summary']})
  logger.write()
  logger.close()


if __name__ == '__main__':
  if check_vscode_interactive():
    _args = [
      "--logroot=logs",
      "--expname=0",
    ]
  else:
    _args = sys.argv[1:]
  config = load_config_from_yaml(pathlib.Path(__file__).parent.parent / 'nprl' / 'configs.yaml', _args)
  logdir = pathlib.Path(config.logdir)
  run(config)

