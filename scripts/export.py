"""
File: export.py
Author: Viet Nguyen
Date: 2025-06-11

Description: This file load a config from a logdir, build expert agent, simulate the
  expert agent, and export the dataset to the dataset folder.
Export dataset from agent. We want to use our trained
  expert agent to export the expert experience to the hdf5 file.
"""

# %%

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from typing import Dict, Optional, List, Any, Tuple
import numpy as np
import os
import h5py
from functools import partial as bind

from lib import Config
from lib.common.config import load_config_from_yaml, load_config_from_raw_yaml
from lib.utils import check_vscode_interactive
from lib.envs.build import build_env, build_expert_dataset, wrap_env
from lib.common import print
from lib.nlp.text_processor import TextProcessor, build_text_processor
from lib.common.counter import Counter
from lib.common.checkpoint import Checkpoint

from nprl.agents.build import build_agent
from nprl import callbacks
from nprl.driver import Driver


def export_episode(ep: Dict[str, np.ndarray], episode_counter: Counter, num_episodes: int, target_dataset_dir: pathlib.Path):
  """
  Export episode data to an HDF5 file.

  Args:
    ep: Dictionary containing episode data
    episode_counter: Counter tracking the current episode number
    target_dataset_dir: Directory to save the dataset file

  Returns:
    The original episode dictionary
  """
  if episode_counter >= num_episodes:
    return ep

  # Get the dataset file path
  h5_path = target_dataset_dir / "dataset.h5"
  episode_idx = int(episode_counter)

  try:
    # Open in append mode if file exists, otherwise create
    file_mode = 'a' if h5_path.exists() else 'w'

    with h5py.File(h5_path, file_mode) as f:
      # Create a group for this episode
      group_name = f"episode_{episode_idx:06d}"

      # Delete group if it already exists (should not happen in normal operation)
      if group_name in f:
        del f[group_name]

      ep_group = f.create_group(group_name)

      # Write each key-value pair in the episode dictionary to the group
      for key, value in ep.items():
        if isinstance(value, np.ndarray):
          ep_group.create_dataset(key, data=value, compression="gzip", chunks=True)
        else:
          # Handle non-numpy data
          dt = h5py.special_dtype(vlen=str) if isinstance(value, str) else None
          ep_group.create_dataset(key, data=np.array([value]), dtype=dt)

    print(f"Exported episode {episode_idx} to {h5_path}", color="green")
  except Exception as e:
    print(f"Error exporting episode {episode_idx}: {e}", color="red")

  return ep


def run(config: Config, expert_logdir: str, target_dataset_dir: str, num_episodes: int):

  expert_logdir = pathlib.Path(expert_logdir)
  target_dataset_dir = pathlib.Path(target_dataset_dir)
  os.makedirs(target_dataset_dir, exist_ok=True)
  step = Counter()

  episode_counter = Counter()

  if config.text_processor:
    text_processor: TextProcessor = build_text_processor(config)
  else:
    text_processor = None

  agent = build_agent(config, text_processor)

  # Checkpoint processing
  checkpoint = Checkpoint(expert_logdir / 'checkpoint.ckpt')
  checkpoint.step = step
  checkpoint.agent = agent
  ckpt_path = expert_logdir / 'checkpoint.ckpt'
  assert ckpt_path.exists(), f"Checkpoint file {ckpt_path} does not exist"
  checkpoint.load(ckpt_path)

  #### Driver Setup and Simulation Loop
  env_fns = [bind(build_env, config, i, text_processor=text_processor,
    multi_task=config.multi_task, reward_shaping=False) for i in range(config.n_envs)]
  driver = Driver(env_fns, parallel=not config.run.debug)
  obs_space = driver.get_field('obs_space')[0]
  act_space = driver.get_field('act_space')[0]

  def episode_counter_increment(ep, _):
    episode_counter.increment()
    return ep
  # driver.on_episode(lambda ep, worker: callbacks.lookback_on_episode(ep, config))
  driver.on_episode(lambda ep, worker: {k: v for k, v in ep.items() if k in (*obs_space.keys(), *act_space.keys())})
  driver.on_episode(lambda ep, worker: export_episode(ep, episode_counter, num_episodes, target_dataset_dir))
  driver.on_episode(episode_counter_increment)

  # Start evaluation loop
  print('Start exporting loop', color='green')

  # Create dataset file and add metadata
  h5_path = target_dataset_dir / "dataset.h5"
  with h5py.File(h5_path, 'w') as f:
    # Add metadata about the dataset
    f.attrs['num_episodes'] = num_episodes
    f.attrs['date_created'] = np.string_(np.datetime64('now').astype(str))

  # Initialize the driver with the policy
  policy = lambda *args: agent.policy(*args, mode='eval')
  # def modifier(trans):
  #   # this is not an expert action, so you cannot self imitate, self_imitate variables
  #   # is only reserved for expert action in the prior preference data
  #   can_imitate = np.asarray([False for i in range(config.n_envs)], dtype=bool) # NOTE: make sure this has the same shape as action
  #   is_expert = np.asarray([False for i in range(config.n_envs)], dtype=bool) # NOTE: make sure this has the same shape as action
  #   external_info = {"is_expert": is_expert, "can_self_imitate": can_imitate}
  #   return {**trans, **external_info}

  # Simulation loop
  driver.reset(agent.init_policy)
  while episode_counter < num_episodes:
    driver(policy, steps=10)

  # Print summary
  print(f"Export complete: {episode_counter.value} episodes saved to {h5_path}", color="green")


if __name__ == '__main__':
  if check_vscode_interactive():
    _args = [
      "../logs_experts/dreamer-metaworld-assembly-v2-expert",
      "../data/metaworld/assembly-v2/data.h5",
      "10"
    ]
  else:
    _args = sys.argv[1:]
  expert_logdir = _args[0]
  target_dataset_dir = _args[1]
  num_episodes = int(_args[2])
  config = load_config_from_raw_yaml(pathlib.Path(expert_logdir) / 'config.yaml')
  run(config, expert_logdir, target_dataset_dir, num_episodes)

