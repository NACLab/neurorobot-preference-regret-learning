"""
File: train.py
Author: Viet Nguyen
Date: 2025-03-18

Description: Main run file for the training/validating any agent in this project
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

from nprl.agents.build import build_agent
from nprl import callbacks
from nprl.utils import EpisodeAggregator
from nprl.driver import Driver

def run(config: Config):

  logger: Logger = build_logger(config)
  logdir = pathlib.Path(config.logdir)
  step = logger.step
  usage = common.Usage(**config.run.usage)
  # train_agg = common.Agg()
  train_normal_agg = common.Agg()
  train_positive_agg = common.Agg()
  val_agg = common.Agg()
  epstats = common.Agg()
  episodes = collections.defaultdict(common.Agg)
  policy_fps = common.FPS()
  train_fps = common.FPS()

  batch_steps = (config.batch_length * config.consec_train + config.replay_context) * config.batch_size
  print(f"[run] batch_steps: {batch_steps}", color='red')
  should_log = when.Clock(config.run.log_every)
  should_report = when.Clock(config.run.report_every)
  should_train = when.Ratio(config.run.train_ratio / batch_steps)
  should_report = when.Clock(config.run.report_every)
  should_save = when.Clock(config.run.save_every)

  @common.timer.section('logfn')
  def logfn(tran, worker):
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add(f'score/{worker}', tran['reward'], agg='sum')
    # print(f"[logfn] reward: {tran['reward']}", color='yellow')
    episode.add(f'length/{worker}', 1, agg='sum')
    episode.add(f'rewards/{worker}', tran['reward'], agg='stack')
    # if 'success_rate' in tran:
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
      }, prefix='episode')
      rew = result.pop(f'rewards/{worker}')
      if len(rew) > 1:
        result[f'reward_rate/{worker}'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      epstats.add(result)

  if config.text_processor:
    text_processor: TextProcessor = build_text_processor(config)
  else:
    text_processor = None

  agent = build_agent(config, text_processor)
  train_carry = [agent.init_train(config.batch_size), agent.init_train(config.batch_size)]
  report_carry = agent.init_report(config.batch_size)
  replay: Replay = build_replay(config, "replay", mode="train")
  positive_replay: Replay = build_replay(config, "positive_replay", mode="train")
  train_dataset = iter(agent.stream(build_stream(config, replay, "train")))
  positive_train_dataset = iter(agent.stream(build_stream(config, positive_replay, "train")))
  val_dataset = iter(agent.stream(build_stream(config, replay, "report")))

  def trainfn(tran, worker):
    needed = config.batch_size * config.batch_length
    if len(replay) < needed:
      # print(f"Replay buffer does not have enough experiences (replay length = {len(replay)} < {needed}), skipping training", color='yellow', end='\r')
      return

    for _train_step in range(should_train(step)):
      # Perform one training step with normal dataset
      with common.timer.section('stream_next'):
        batch = next(train_dataset) # Half train with normal dataset
      train_carry[0], outs_normal, train_mets_normal = agent.train(train_carry[0], batch)
      if 'replay' in outs_normal:
        replay.update(outs_normal['replay'])
      train_normal_agg.add({**train_mets_normal}, prefix='train_normal')

      # Perform one training step with positive dataset
      if len(positive_replay) < needed and config.use_imiln:
        # print(f"Positive replay buffer does not have enough experiences (positive replay length = {len(positive_replay)} < {needed}), skipping training", color='yellow', end='\r')
        pass
      elif config.use_imiln:
        with common.timer.section('stream_next_positive'):
          batch = next(positive_train_dataset) # Half train with positive dataset
        train_carry[1], outs_positive, train_mets_positive = agent.train(train_carry[1], batch)
        if 'replay' in outs_positive:
          positive_replay.update(outs_positive['replay'])
        train_positive_agg.add({**train_mets_positive}, prefix='train_positive')

      # print(f"Training Step {_train_step + 1}", end='\r', color='green')
      train_fps.step(batch_steps)


  # Checkpoint processing
  if config.run.save_every > 0:
    should_save = when.Clock(config.run.save_every)
    checkpoint = Checkpoint(logdir / 'checkpoint.ckpt')
    checkpoint.step = step
    checkpoint.agent = agent
    if config.run.from_checkpoint:
      checkpoint.load(config.run.from_checkpoint)
    checkpoint.load_or_save()
    should_save(step)  # Register that we just saved.

  # Loading the replay buffer if needed, we might have collected some experience in the past
  replay.load()
  positive_replay.load()

  #### Driver Setup and Simulation Loop
  env_fns = [bind(build_env, config, i, text_processor=text_processor,
    multi_task=config.multi_task) for i in range(config.n_envs)]
  driver = Driver(env_fns, parallel=not config.run.debug)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  # driver.on_step(replay.add) # we add experience at the end of each episode
  driver.on_step(logfn)
  driver.on_step(trainfn)
  driver.on_episode(lambda ep, worker: callbacks.lookback_on_episode(ep, config))
  driver.on_episode(lambda ep, worker: callbacks.add_trajectory(ep, replay, positive_replay, worker, config.use_selfrefl))

  # Start training and simulation loop
  print('Start training loop', color='green')

  # Initialize the driver with the policy
  policy = lambda *args: agent.policy(*args, mode='train')
  def modifier(trans):
    can_imitate = np.asarray([False for i in range(config.n_envs)], dtype=bool)
    is_expert = np.asarray([False for i in range(config.n_envs)], dtype=bool)
    external_info = {"is_expert": is_expert, "can_self_imitate": can_imitate}
    return {**trans, **external_info}

  # Simulation loop
  driver.reset(agent.init_policy)
  while step < config.run.steps:
    driver(policy, steps=10, modifier=modifier)

    if should_report(step) and len(replay):
      print('Evaluation')
      report_agg = common.Agg()
      for _ in range(config.run.report_batches):
        batch = next(val_dataset)
        report_carry, mets = agent.report(report_carry, batch)
        report_agg.add(mets)
      logger.add(report_agg.result(), prefix='report')

    # Logging things
    if should_log(step):
      # logger.add(train_agg.result())
      logger.add(train_normal_agg.result())
      logger.add(train_positive_agg.result())
      logger.add(val_agg.result())
      logger.add(epstats.result(), prefix='epstats')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.add({'timer': common.timer.stats()['summary']})
      logger.write()

    if should_save(step):
      checkpoint.save()

  # Save the checkpoint at the end of the training
  checkpoint.save()

  # Logging one last time
  logger.add(train_normal_agg.result())
  logger.add(train_positive_agg.result())
  logger.add(val_agg.result())
  logger.add(epstats.result(), prefix='epstats')
  logger.add(usage.stats(), prefix='usage')
  logger.add({'fps/policy': policy_fps.result()})
  logger.add({'fps/train': train_fps.result()})
  logger.add({'timer': common.timer.stats()['summary']})
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

