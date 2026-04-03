"""
File: collect.py
Author: Viet Nguyen
Date: 2025-03-18

Description: Main run file for collecting expert trajectories.
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
from lib.nlp.text_processor import TextProcessor, build_text_processor
from lib.envs.build import build_env, build_expert_dataset, wrap_env
from lib.replay import build_replay
from lib.replay.replay import Replay
from lib import streams
from lib.common.checkpoint import Checkpoint
from lib import common
from lib.common import print

from lib.agent.active import JAXAgent
from nprl.agents.build import build_agent
from nprl import callbacks
from nprl.utils import EpisodeAggregator

def run(config: Config):

  logger: Logger = build_logger(config, prefix='expert')
  expert_step = logger.step
  batch_steps = (config.batch_length * config.consec_train + config.replay_context) * config.batch_size
  print(f"[run] batch_steps: {batch_steps}", color='red')

  text_processor: TextProcessor = build_text_processor(config)
  replay: Replay = build_replay(config, "replay", mode="train")
  positive_replay: Replay = build_replay(config, "positive_replay", mode="train", force_offline=True)
  agent: JAXAgent = build_agent(config, text_processor)

  episodes = collections.defaultdict(common.Agg)
  epstats = common.Agg()

  def logfn(tran, worker):
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add(f'score/{worker}', tran['reward'], agg='sum')
    for key, value in tran.items():
      value = np.asarray(value)
      if value.dtype == np.uint8 and value.ndim == 3:
        if worker == 0:
          episode.add(f'policy_{key}', value, agg='stack')

    if tran['is_last']:
      result = episode.result()
      logger.add({
          f'score/{worker}': result.pop(f'score/{worker}'),
      }, prefix='expert_episode')
      epstats.add(result)

  if config.expert.enable:
    episode_agg = EpisodeAggregator()
    print(f"[run] [expert] filling expert trajectories", color='green')
    expert_dataset = build_expert_dataset(config, text_processor=text_processor)
    expert_env = wrap_env(expert_dataset.env, config)
    it = iter(expert_dataset.dataset(expert_env))
    episode_id = 0
    while episode_id < config.expert.n_episodes:
      transition = next(it)
      external_data = {'is_expert': np.asarray(True), 'can_self_imitate': np.asarray(True)}
      _transition = {k: v for k, v in transition.items() if not k.startswith('_')}
      for k, s in agent.ext_space.items():
        if k not in external_data and k not in ('consec', 'stepid'):
          v = np.zeros(s.shape, dtype=s.dtype)
          external_data[k] = v
      final_trn = {**_transition, **external_data}
      # NOTE: Workaround for missing success_rate and relative_stability in some environments
      if 'relative_stability' not in final_trn:
        final_trn['relative_stability'] = np.asarray(0.0)
      if 'success_rate' not in final_trn:
        final_trn['success_rate'] = np.asarray(1.0)
      episode_agg.add(final_trn)
      logfn(final_trn, 0)
      expert_step.increment()
      if transition['is_last'] or transition['is_terminal']:
        ep = episode_agg.result()
        ep = callbacks.lookback_on_episode(ep, config)
        ep = callbacks.add_trajectory(ep, replay, positive_replay, 0)
        # Finally, write log
        logger.add(epstats.result(), prefix='expert_epstats')
        logger.write()
        episode_id += 1
    replay.save()
    positive_replay.save()
  print("Finished collecting expert trajectories", color='green')


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

