
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import re
import functools
import cv2
from typing import List

import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)

from ..base import Env, ExpertDataset
from ...common import Space, print
from ...envs.from_env.gym import GymEnv
from ...nlp.text_processor import TextProcessor, DefaultTextProcessor
from ...common import tree


# NOTE: Viet Dung Nguyen: Currently, we support the following tasks
INSTRUCTIONS = {
  # "bottom burner": (
  #   "Turn the oven knob that activates the bottom burner",
  #   "Activates the bottom burner by turning the knob",
  #   "Turn the knob around to activate the bottom burner",
  #   "Please turn the knob around to activate the bottom burner"
  # ),
  # "top burner": (
  #   "Turn the oven knob that activates the top burner",
  #   "Activates the top burner by turning the knob",
  #   "Turn the knob around to activate the top burner",
  #   "Please turn the knob around to activate the top burner"
  # ),
  "light switch": (
    "Turn the light switch on",
    "Turn on the light switch",
    "Please turn on the light switch"
  ),
  "slide cabinet": (
    "Open the right sliding cabinet",
    "Open the slide cabinet",
    "Please slide the right cabinet open"
  ),
  # "hinge cabinet": (
  #   "Open the left hinge cabinet",
  #   "Please open the left cabinet",
  #   "Please open the left hinge cabinet"
  # ),
  "microwave": (
    "Open the microwave door",
    "Open the microwave",
    "Please open the microwave",
    "Please open the microwave door"
  ),
  "kettle": (
    "Move the kettle to the top left",
    "Move the kettle to the top left burner",
    "Please move the kettle to the top left",
    "Please move the kettle to the top left burner"
  )
}


class KitchenEnv(GymEnv):
  def __init__(self, task: str | List[str], text_processor: TextProcessor = None,
      image_size = 128, obs_render: bool = False, img_obs_key: str = 'image',
      object_noise_ratio: float = 0.0005, robot_noise_ratio: float = 0.01,
      external_env: gym.Env | None = None, max_episode_steps: int = 280, **kwargs):
    """
    This is the main class for the kitchen environment. There are three modes:
    1. Single-task mode: The agent will be given a single task to complete. There will be no reward for other tasks.
    2. Multi-task combined mode: The agent will be given a list of tasks to complete. The agent has to do all the task in one run.
    3. Multi-task mode: The agent will be given a list of tasks, but it only has to do each task in a separate episode.

    Args:
        task (str | List[str]): The task to be completed. If it is a string, it will be used as the current task (in the single-task mode).
          if task name is `combined`, it will be used as the multi-task mode, but the agent has to do all the task in one run.
          If it is a list, it will be used as the tasks list to be completed (each task is for each episode).
        text_processor (TextProcessor, optional): The text processor to be used. Defaults to None.
        image_size (int, optional): The size of the image. Defaults to 128.
        obs_render (bool, optional): Whether to render the observations. Defaults to False.
        img_obs_key (str, optional): The key of the image observation. Defaults to 'image'.
        object_noise_ratio (float, optional): The noise ratio of the objects. Defaults to 0.0005.
        robot_noise_ratio (float, optional): The noise ratio of the robot. Defaults to 0.01.
        external_env (gym.Env | None, optional): The external environment to be used instead of the default created environment. Defaults to None.
        max_episode_steps (int, optional): The maximum number of steps per episode. Defaults to 280.
    """

    self._combined = False
    if isinstance(task, str) and task == 'combined':
      task = ['kettle', 'light switch', 'microwave', 'slide cabinet']
      self._combined = True

    tasks = [task] if isinstance(task, str) else task

    self.external_env = external_env
    if external_env is None:
      env = gym.make('FrankaKitchen-v1',
        tasks_to_complete=tasks,
        render_mode='rgb_array',
        terminate_on_tasks_completed=False,
        object_noise_ratio=object_noise_ratio,
        robot_noise_ratio=robot_noise_ratio,
        max_episode_steps=max_episode_steps,
        **kwargs
      )
    else:
      env = external_env

    super().__init__(env, img_obs_key=img_obs_key, obs_render=obs_render, image_size=image_size)

    # assert obs_render is False, "obs_render is not supported for the kitchen environment because there is no recorded expert data for that, one can comment out this line and use the image as observation."
    self.tasks = tasks
    self.text_processor = text_processor if text_processor is not None else DefaultTextProcessor(max_sequence_length=32)
    self.image_size = image_size

    self._reset()

  def _reset(self):
    # Reward handling
    # the actual gym-robotics reward exists at the event of task completion, not the number of
    # completed tasks up until that point, so we need to modify the reward
    # to do so, in order to align with the reward function of the original
    # environment.
    self.reward = np.zeros(())
    self.current_task_id = np.random.randint(0, len(self.tasks))
    self.current_task = self.tasks[self.current_task_id]
    self.reset_task_text()
    self.prev_reward_for_checking_success = np.zeros(())

  @functools.cached_property
  def obs_space(self):
    spaces = super().obs_space
    new_spaces = {}
    for key, value in spaces.items():
      if key.startswith('achieved_goal'):
        _key = f"ag_{key[len('achieved_goal')+1:].replace(' ', '')}"
        new_spaces[_key] = value
      elif key.startswith('desired_goal'):
        _key = f"dg_{key[len('desired_goal')+1:].replace(' ', '')}"
        new_spaces[_key] = value
      else:
        new_spaces[key] = value
    return {
      **new_spaces,
      'instruction_input_ids': Space(shape=(self.text_processor.max_sequence_length,), dtype=np.int32, low=0, high=self.text_processor.vocab_size - 1),
      'instruction_attention_mask': Space(shape=(self.text_processor.max_sequence_length,), dtype=bool),
    }

  def reset_task_text(self):
    # Base on the current task, reset the task text
    _ins = [INSTRUCTIONS[task] for task in self.tasks]
    _ins = [tup[np.random.randint(0, len(tup))] for tup in _ins]
    if self._combined:
      self.current_task_text = ", ".join(_ins)
    else:
      self.current_task_text = _ins[self.current_task_id]
    _out = self.text_processor.encode(self.current_task_text)
    self.instruction_input_ids = _out['input_ids'][0]
    self.instruction_attention_mask = _out['attention_mask'][0]

  # To be used in the callbacks of MeasureSuccessRate wrapper
  def is_success(self, obs):
    # return obs['reward'] > 0 # NOTE: Consider this
    if self._combined:
      # return self.reward >= len(self.tasks)
      success = False
      if self.reward > self.prev_reward_for_checking_success:
        success = True
      self.prev_reward_for_checking_success = self.reward
      return success
    else:
      return self.reward > 0

  def step(self, action):

    out = super().step(action)

    new_out = {}
    for key, value in out.items():
      if key.startswith('achieved_goal'):
        _key = f"ag_{key[len('achieved_goal')+1:].replace(' ', '')}"
        new_out[_key] = value
      elif key.startswith('desired_goal'):
        _key = f"dg_{key[len('desired_goal')+1:].replace(' ', '')}"
        new_out[_key] = value
      else:
        new_out[key] = value

    if self._combined:
      self.reward = np.asarray(len(self._env.unwrapped.episode_task_completions), dtype=np.float32)
    else:
      if self.current_task in self._env.unwrapped.episode_task_completions:
        self.reward = np.asarray(1.0, dtype=np.float32)

    new_out = {**new_out, 'reward': self.reward, 'instruction_input_ids': self.instruction_input_ids,
      'instruction_attention_mask': self.instruction_attention_mask}

    # if this is external env, we just use the reward from the external env
    if self.external_env:
      new_out['reward'] = out['reward']

    if new_out['is_terminal'] or new_out['is_last']:
      self._reset()

    return new_out


class KitchenEnvSimulatorFromDataset(ExpertDataset):
  def __init__(self, task: str | List[str], text_processor: TextProcessor = None, seed: int = None, image_size = 192,
      dataset_name: str = 'D4RL/kitchen/complete-v2', n_episodes: int = 10, obs_render: bool = False,
      img_obs_key: str = 'image', object_noise_ratio: float = 0.0005, prockeys: str = r'.*',
      robot_noise_ratio: float = 0.01, max_episode_steps: int = 280, **kwargs):
    import minari
    self._n_episodes = n_episodes
    assert dataset_name == 'D4RL/kitchen/complete-v2'
    self._dataset = minari.load_dataset(dataset_name)
    self._dataset.set_seed(seed=seed if seed is not None else np.random.randint(np.iinfo(np.int32).max))
    # For computing rewards
    # PREDEFINED_TASKS = {
    #   'D4RL/kitchen/complete-v2': ['kettle', 'light switch' 'microwave', 'slide cabinet'],
    #   'D4RL/kitchen/mixed-v2': ['bottom burner', 'microwave', 'kettle', 'light switch'],
    #   'D4RL/kitchen/partial-v2': ['light switch', 'kettle', 'microwave', 'slide cabinet'],
    # }[dataset_name]
    # All dataset has these goals: https://minari.farama.org/datasets/D4RL/kitchen/
    # PREDEFINED_TASKS = ['kettle', 'light switch', 'microwave', 'slide cabinet']
    PREDEFINED_TASKS = "combined" # ['kettle', 'light switch', 'microwave', 'slide cabinet']
    self._orig_env = self._dataset.recover_environment(render_mode='rgb_array')
    self._env = KitchenEnv(PREDEFINED_TASKS, text_processor, image_size, obs_render=obs_render, img_obs_key=img_obs_key,
      external_env=self._orig_env, object_noise_ratio=object_noise_ratio, robot_noise_ratio=robot_noise_ratio,
      max_episode_steps=max_episode_steps, **kwargs)
    self.image_size = image_size
    self._obs_render = obs_render

    predefine_tasks = ['kettle', 'light switch', 'microwave', 'slide cabinet']
    self.taken_keys = [k for k in predefine_tasks if re.match(prockeys, k)]

  @functools.cached_property
  def env(self) -> Env:
    # Wrapped env, which already has action unnormalization. E.g., agent -> [-1, 1] -> unnormalize -> [-3, 6] -> env
    return self._env

  def dataset(self, final_env: Env):
    episodes = self._dataset.sample_episodes(n_episodes = self._n_episodes)
    epid = 0
    while True:
      episode = episodes[epid]
      print(f"[run] [expert] fill_count (ep): {epid} / {len(episodes)}", end='\r', color='green')
      expert_actions = episode.actions
      expert_observation = episode.observations['observation']
      expert_ag = episode.observations['achieved_goal']
      expert_dg = episode.observations['desired_goal']
      expert_rewards = episode.rewards
      expert_terminals = episode.terminations
      expert_lasts = episode.truncations | episode.terminations
      episode_length = len(expert_actions)
      prev_reward = 0
      has_succeeded = False
      if self._obs_render: # NOTE: we add the image from the env because the dataset does not have the image
        final_env_obs = final_env.step({'action': final_env.act_space['action'].sample(), 'reset': True})
      for i in range(episode_length):
        achieved_goal = {
          f'ag_{k.replace(" ", "")}': np.asarray(v[i], dtype=np.float32) for k, v in expert_ag.items() if k in self.taken_keys
        }
        desired_goal = {
          f'dg_{k.replace(" ", "")}': np.asarray(v[i], dtype=np.float32) for k, v in expert_dg.items() if k in self.taken_keys
        }
        reward = expert_rewards[i]
        is_success = False
        if reward > prev_reward:
          is_success = True
          prev_reward = reward
        if reward > 0:
          has_succeeded = True
        _action = final_env.act_space['action'].normalize(expert_actions[i])
        transition = {
          'observation': expert_observation[i],
          'action': _action,
          'reward': expert_rewards[i],
          'is_terminal': expert_terminals[i],
          'is_last': expert_lasts[i] or i == episode_length - 1,
          **achieved_goal,
          **desired_goal,
          'has_succeeded': has_succeeded,
          'is_success': is_success,
          'relative_stability': 0.0,
          'is_first': i == 0,
          'success_rate': 1.0,
          'instruction_input_ids': self._env.instruction_input_ids,
          'instruction_attention_mask': self._env.instruction_attention_mask
        }
        if self._obs_render:
          transition['image'] = final_env_obs['image'] # we record the current image first, then step
          final_env_obs = final_env.step({'action': _action, 'reset': False})
        yield transition
      epid = (epid + 1) % len(episodes)


  def close(self, final_env: Env):
    self._orig_env.close()
    final_env.close()
    self._env.close()

