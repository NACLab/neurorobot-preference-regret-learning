from .base import *
from .build import *
from . import wrappers

import gymnasium as gym
gym.register("mtc", "lib.envs.mtc:MountainCarContinuousImageObservation")

