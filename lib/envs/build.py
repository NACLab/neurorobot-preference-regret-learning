"""
File: build.py
Author: Viet Nguyen
Date: 2025-03-18

Description: Build the environment
"""


import importlib
import pathlib
import os

from ..common import print
from . import wrappers
from .base import ExpertDataset
from ..nlp.text_processor import TextProcessor


def build_env(config, index, text_processor: TextProcessor | None = None, multi_task: bool = False, **overrides):
  if multi_task:
    suite, tasks = config.suite, config.tasks
  else:
    suite, task = config.task.split('_', 1)

  ## NOTE: ENV_VARS setup: if we are training on RIt's Research Computing
  # Cluster, if so, we have to change several environment variables such as mujoco_gl engine
  # we only need to change this variables when training in gymrobotics environment on RC
  if "rc" in config and config.rc and (suite == "gymrobotics" or suite == "metaworld" or suite =='kitchen'):
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["MUJOCO_GL"] = "egl"

  ctor = {
    'gym': 'lib.envs.from_env.gym:GymEnv',
    'mtc': 'lib.envs.mtc.mtc:MountainCarGeneralEnv',
    'kitchen': 'lib.envs.kitchen.kitchen:KitchenEnv',
    'robosuite': 'lib.envs.robosuite.robosuite:RobosuiteEnv',
    'languagetable': 'lib.envs.language_table.language_table:LanguageTableEnv',
    'metaworld': 'lib.envs.metaworld.metaworld:MetaWorldEnv',
    'qenv': 'lib.envs.qenv.qenv:QueueEnv'
  }[suite]
  if isinstance(ctor, str):
    module, cls = ctor.split(':')
    module = importlib.import_module(module)
    ctor = getattr(module, cls)
  kwargs = config.env.get(suite, {})
  kwargs.update(overrides)
  if kwargs.pop('use_seed', False):
    kwargs['seed'] = hash((config.seed, index)) % (2 ** 32 - 1)
  if kwargs.pop('use_logdir', False):
    kwargs['logdir'] = pathlib.Path(config.logdir) / f'env{index}'
  text_processor_kw = {'text_processor': text_processor} if text_processor is not None else {}
  # assert text_processor is not None, "text_processor is required"
  if text_processor is None:
    print(f"[WARNING] text_processor was not used", color="yellow")
  env = ctor(tasks if multi_task else task, **kwargs, **text_processor_kw)
  return wrap_env(env, config)


def wrap_env(env, config):
  for name, space in env.act_space.items():
    if not space.discrete:
      env = wrappers.NormalizeAction(env, name)
  env = wrappers.UnifyDtypes(env)
  env = wrappers.CheckSpaces(env)
  for name, space in env.act_space.items():
    if not space.discrete:
      env = wrappers.ClipAction(env, name)
  # Environment measurement
  env = wrappers.EnvStepLambdaWrapper(env, wrappers.MeasureSuccessRate(env))
  env = wrappers.EnvStepLambdaWrapper(env, wrappers.MeasureRStability(100, 100, reward_offset=config.rstability_reward_offset))
  return env


def build_expert_dataset(config, text_processor: TextProcessor, multi_task: bool = False, **overrides) -> ExpertDataset:
  # suite, task = config.task.split('_', 1)
  # For multi-task
  if multi_task:
    suite, tasks = config.suite, config.tasks
  else:
    suite, task = config.task.split('_', 1)

  # NOTE: handle edge case for robosuite
  if suite == 'robosuite' and task in ('Lift', 'Door'): # handle edge case for robosuite
    _suite = 'robosuite_common'
  else:
    _suite = suite

  ctor = {
    'kitchen': 'lib.envs.kitchen.kitchen:KitchenEnvSimulatorFromDataset',
    'robosuite': 'lib.envs.robosuite.robosuite:RobosuiteExpertDataset',
    'languagetable': 'lib.envs.language_table.language_table:LanguageTableExpertDataset',
    'mtc': 'lib.envs.mtc.mtc:MountainCarExpertDataset',
    'metaworld': 'lib.envs.expert_dataset:CommonExpertDataset',
    'robosuite_common': 'lib.envs.expert_dataset:CommonExpertDataset',
    'qenv': 'lib.envs.expert_dataset:CommonExpertDataset'
  }[_suite]

  # env kwargs
  env_kwargs = config.env.get(suite, {}) # environment suite
  expert_kwargs = config.expert.get(_suite, {}) # expert suite, can have multiple env suite
  kwargs = {**env_kwargs, **expert_kwargs}
  kwargs.update(overrides)
  if isinstance(ctor, str):
    module, cls = ctor.split(':')
    module = importlib.import_module(module)
    ctor = getattr(module, cls)
  return ctor(tasks if multi_task else task, text_processor=text_processor, seed=config.seed, **kwargs)

