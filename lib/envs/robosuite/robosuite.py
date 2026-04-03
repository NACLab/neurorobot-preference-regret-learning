import numpy as np
import functools
from typing import List, Tuple, Dict
import platform
import os
import warnings
import sys
import io
import re
import cv2
import h5py
import pathlib

from ..base import Env, ExpertDataset
from ...common import Space, print
from ...nlp.text_processor import TextProcessor

try:
  # robosuite.ALL_ENVIRONMENTS
  # from mimicgen import mimicgen
  import mimicgen
except ImportError:
  print("Mimicgen not found, you might want to install mimicgen for more robosuite environments")


SIMPLE_OBS_KEYS = [
  'agentview_image',
  'object',
  'robot0_eef_pos',
  'robot0_eef_quat',
  'robot0_eye_in_hand_image',
  'robot0_gripper_qpos',
  'robot0_gripper_qvel',
  'robot0_joint_pos_cos',
  'robot0_joint_pos_sin',
  'robot0_joint_vel'
]

IMITATION_DATASET_MAPPING = {
  'Coffee_D0': 'coffee.hdf5',
  'CoffeePreparation_D0': 'coffee_preparation.hdf5',
  'Threading_D0': 'threading.hdf5',
  'ThreePieceAssembly_D0': 'three_piece_assembly.hdf5',
  'MugCleanup_D0': 'mug_cleanup.hdf5',
  'Stack_D0': 'stack.hdf5',
  'StackThree_D0': 'stack_three.hdf5',
  'NutAssembly_D0': 'nut_assembly.hdf5',
  'Square_D0': 'square.hdf5',
  'PickPlace_D0': 'pick_place.hdf5',
}

ROBOT_MAPPING = {
  'Coffee_D0': ['Panda'],
  'CoffeePreparation_D0': ['Panda'],
  'Threading_D0': ['Panda'],
  'ThreePieceAssembly_D0': ['Panda'],
  'MugCleanup_D0': ['Panda'],
  'Stack_D0': ['Panda'],
  'StackThree_D0': ['Panda'],
  'NutAssembly_D0': ['Sawyer'],
  'Square_D0': ['Panda'],
  'PickPlace_D0': ['Sawyer'],
}



# class RegexFilteredOutput(io.StringIO):
#   def __init__(self, regex_pattern, *args, **kwargs):
#     super().__init__(*args, **kwargs)
#     self.regex_pattern = re.compile(regex_pattern)

#   def write(self, s):
#     if not self.regex_pattern.search(s):
#       super().write(s)
# # Define regex pattern
# regex_pattern = r'Joint limit reached in joint.*'
# # Create a StringIO object with the regex filter
# filtered_output = RegexFilteredOutput(regex_pattern)
# # Redirect stdout to the filtered output
# sys.stdout = filtered_output

class RobosuiteEnv(Env):
  def __init__(self, task: str | object,
      robots: List[str] = ["Panda"],
      gripper_types: str = "default",
      controller: str = "OSC_POSE",
      env_config: str = "default",
      image_size: Tuple[int, int] = (96, 96), # width, height
      camera_names: List[str] = ["agentview", "sideview", "frontview", "birdview", "robot0_eye_in_hand"],
      reward_shaping: bool = False,
      generate_images=True,
      has_renderer=False,
      has_offscreen_renderer=True,
      control_freq=20,
      controller_config = None,
      simple_obs = False,
      text_processor: TextProcessor | None = None,
      render_image_size: Tuple[int, int] = (200, 200),
      max_episode_steps: int = 1000,
      use_84x84_images: bool = True, # This is a workaround because the dataset is only available in 84x84 images
      **kwargs):
    """

    Args:
        task (str): Can be one in these string:
          - Single Arm Env: Door, Lift, NutAssembly, PickPlace, Stack, ToolHang, Wipe
          - Two Arm Env: TwoArmHandover, TwoArmLift, TwoArmPegInHole, TwoArmTransport
        robots (List[str], optional): _description_. Defaults to ["Panda"]. Robots:
          SingleArm: IIWA, Jaco, Kinova3, Panda, Sawyer, UR5e
          Bimanual: Baxter
        gripper_types (str, optional): _description_. Defaults to "default". Gripper types:
          defualt, PandaGripper, RethinkGripper, ...?
        controller (str, optional): _description_. Defaults to "OSC_POSE". controllers:
          Position-based control: input actions are, by default, interpreted as delta values from the current state.
            `OSC_POSITION`: desired position
            `JOINT_POSITION`: desired joint configuration
            End-effector pose controller: delta rotations from the current end-effector orientation in the form of axis-angle coordinates (ax, ay, az)
              `OSC_POSE`: the rotation axes are taken relative to the global world coordinate frame
                the desired value is the 6D pose (position and orientation) of a controlled frame. We follow the formalism from [Khatib87].
              `IK_POSE`: the rotation axes are taken relative to the end-effector origin, NOT the global world coordinate frame!
          `JOINT_VELOCITY`: action dimension: number of joints
          `JOINT_TORQUE`: action dimension: number of joints
          More information: https://robosuite.ai/docs/modules/controllers.html
        env_config (str, optional): _description_. Defaults to "default". Setup of the two arms, possible values:
          default, single-arm-opposed, single-arm-parallel
        img_size (Tuple[int, int], optional): _description_. Defaults to (128, 128).
        camera_names (List[str], optional): _description_. Defaults to ["agentview", "sideview", "frontview", "birdview"].
          Available cameras: 'frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand', 'robot1_robotview', 'robot1_eye_in_hand', ...
        reward_shaping (bool, optional): _description_. Defaults to False.
    """
    # warning filter
    # warnings.filterwarnings('ignore', 'Joint limit reached in joint 3')
    # when rendering images
    import robosuite

    # Macro setup so that the image is not flipped by default
    # https://robosuite.ai/docs/modules/environments.html#observations
    import robosuite.macros as macros
    macros.IMAGE_CONVENTION = "opencv"

    # load default controller parameters for Operational Space Control (OSC)
    # https://robosuite.ai/docs/modules/controllers.html
    # OSC_POSE, OSC_POSITION, IK_POSE, JOINT_POSITION, JOINT_VELOCITY, or JOINT_TORQUE
    if controller_config is None:
      from robosuite.controllers import load_controller_config
      controller_config = load_controller_config(default_controller=controller)
    else:
      controller_config = controller_config

    # For window machines
    if platform.system() == "Windows":
      os.environ['MUJOCO_GL'] = "osmesa"
      os.environ["PYOPENGL_PLATFORM"] = "osmesa"
      os.environ["MUJOCO_EGL_DEVICE_ID"] = "1" # https://github.com/google-deepmind/dm_control/issues/415#issue-1812075254

    # Wipe env only accept wiping gripper
    gripper_types = 'WipingGripper' if task == 'Wipe' else gripper_types

    # NOTE: PickPlace_D0 has no sideview camera, so remove it to avoid error
    if task == 'PickPlace_D0' or task == 'PickPlace':
      camera_names = [c for c in camera_names if c != 'sideview']

    # Workaround for image size
    if use_84x84_images:
      self.original_image_width = 84
      self.original_image_height = 84
    else:
      self.original_image_width = image_size[0]
      self.original_image_height = image_size[1]
    self.image_width = image_size[0]
    self.image_height = image_size[1]

    # Change the robot (because the dataset is not available for all robots)
    if task in ROBOT_MAPPING:
      robots = ROBOT_MAPPING[task]

    # https://robosuite.ai/docs/modules/environments.html
    # Api docs: https://robosuite.ai/docs/simulation/environment.html?highlight=camera_name#robot-environment
    # create an environment for policy learning from pixels
    if isinstance(task, str):
      self._env = robosuite.make(
        task,
        robots=robots, # load a robot
        gripper_types=gripper_types, # use default grippers per robot arm
        controller_configs=controller_config,# each arm is controlled using OSC
        env_configuration=env_config, # (two-arm envs only) arms face each other:
        has_renderer=has_renderer, # no on-screen rendering
        has_offscreen_renderer=has_offscreen_renderer, # off-screen rendering needed for image obs
        control_freq=control_freq, # 20 hz control for applied actions
        horizon=max_episode_steps, # each episode have maximum of 2000 steps
        use_object_obs=True,
        use_camera_obs=generate_images, # provide image observations to agent
        camera_names=camera_names, # use camera for observations
        camera_heights=self.original_image_height, # image height
        camera_widths=self.original_image_width, # image width
        # camera_depths=True, # potentialy use depth
        # camera_segmentations=?, # potentially use segmentation
        reward_shaping=reward_shaping, # use a dense reward signal for learning
        # renderer_config=renderer_config,
        # renderer='nvisii'
        **kwargs
      )
    else:
      self._env = task
    self._task = task
    self._done = True
    self._reward_shaping = reward_shaping
    self._simple_obs = simple_obs
    self._render_image_size = render_image_size
    self._camera_names = camera_names

    ## NOTE: Making sure that all observation does not have shape 0, otherwise, it will not
    # work with the space lib
    raw_spaces = {}
    should_expand_keys = set()
    # f32_keys = set()
    for k, v in self._env.observation_spec().items():
      if not hasattr(v, 'shape') or v.shape == () or v.shape == (0,):
        raw_spaces[k] = Space(np.float32, (1,))
        should_expand_keys.add(k)
        # f32_keys.add(k)
      elif len(v.shape) == 3:
        raw_spaces[k] = Space(np.uint8, v.shape, low=0, high=255)
      else:
        raw_spaces[k] = Space(np.float32, v.shape)
        # f32_keys.add(k)
    # self.f32_keys = f32_keys

    if self._simple_obs:
      self._allowed = SIMPLE_OBS_KEYS
    else:
      self._allowed = None

    # NOTE: Special case, for the wipe environment, since we are not using the
    # same gripper, the space should be omitted
    if task == "Wipe":
      raw_spaces.pop('robot0_gripper_qpos')
      raw_spaces.pop('robot0_gripper_qvel')

    # Finally, assign them to object variables
    self.raw_spaces = raw_spaces
    self.should_expand_keys = should_expand_keys
    self._resizing_image_space_keys = []
    if self.image_width != self.original_image_width or self.image_height != self.original_image_height:
      for k, v in raw_spaces.items():
        if len(v.shape) == 3 and v.dtype == np.uint8:
          self._resizing_image_space_keys.append(k)

  @functools.cached_property
  def obs_space(self):
    spaces = {}
    for k, s in self.raw_spaces.items():
      if k == 'object-state':
        spaces['object'] = s
      else:
        spaces[k] = s
    if self._simple_obs:
      spaces = {k: v for k, v in spaces.items() if k in self._allowed}
    for k in self._resizing_image_space_keys:
      if k in spaces:
        spaces[k] = Space(np.uint8, (self.image_width, self.image_height, 3), low=0, high=255)
    return {
      **spaces,
      'reward': Space(np.float32),
      'is_first': Space(bool),
      'is_last': Space(bool),
      'is_terminal': Space(bool),
      'success': Space(bool)
    }

  @functools.cached_property
  def act_space(self):
    low, high = self._env.action_spec
    action = Space(np.float32, low.shape, low, high)
    return {"reset": Space(bool), "action": action}

  @functools.cached_property
  def unroll(self):
    return self._env

  def step(self, action):
    action = action.copy()
    reset = action.pop('reset')
    is_first = False
    is_last = False
    if reset or self._done:
      is_first = True
      raw_obs = self._env.reset()
      self._done = False
      reward = np.asarray(0, dtype=np.float32)
    else:
      action = action["action"]
      raw_obs, reward, is_last, _ = self._env.step(action)
      reward = np.asarray(reward, dtype=np.float32)
      self._done = is_last
    # raw_obs = {k: np.asarray(v, dtype=np.float32) if k in self.f32_keys else v for k, v in raw_obs.items()}
    success = self._env._check_success()
    return self.__obs(raw_obs, reward, is_first, is_last, success)

  def __obs(self, raw_obs, reward, is_first, is_last, success):
    # raw_obs here is an ordereddict
    _raw_obs = dict(raw_obs)
    for k in self.should_expand_keys:
      _raw_obs[k] = np.asarray([_raw_obs[k]])
    if 'object-state' in _raw_obs:
      _raw_obs['object'] = _raw_obs['object-state']
      _raw_obs.pop('object-state')
    if self._task == "Wipe":
      _raw_obs.pop("robot0_gripper_qpos")
      _raw_obs.pop("robot0_gripper_qvel")
    if self._simple_obs:
      _raw_obs = {k: v for k, v in _raw_obs.items() if k in self._allowed}
    for k in self._resizing_image_space_keys:
      if k in _raw_obs:
        _raw_obs[k] = cv2.resize(_raw_obs[k], (self.image_width, self.image_height))
    # Process reward if sprase
    # NOTE: try adding 0.1 to it, but for simplicity, we just keep it as normal
    # _reward = np.float64(reward)
    # if not self._reward_shaping: # reward {0, 1} => {0.1, 1.1}
    #   # Reward the agent to live more even when it has not
    #   # succeed => min reward will be 50, but if the environment terminates early,
    #   # it will not even get to 50
    #   _reward += 0.1
    _raw_obs = {k: np.asarray(v, dtype=self.obs_space[k].dtype) for k, v in _raw_obs.items() if k in self.obs_space}
    return dict(
      reward=reward, # _reward NOTE: try adding 0.1 to it, but for simplicity, we just keep it as normal
      is_first=is_first,
      is_last=is_last,
      is_terminal=is_last,
      success=success,
      **_raw_obs,
    )

  def is_success(self, trn: Dict[str, np.ndarray]):
    return self._env._check_success()

  def render(self):
    # We have to flip the image ([::-1]) because the image is flipped by default opengl convention
    return {view: self._env.sim.render(
      camera_name=view,
      width=self._render_image_size[0],
      height=self._render_image_size[1],
      depth=False,
    )[::-1] for view in self._camera_names}

class RobosuiteExpertDataset(ExpertDataset):

  def __init__(self, task: str, text_processor: TextProcessor | None = None, seed: int | None = None, *,
      dataset_dir: str | None = None, **env_kwargs):

    import robomimic
    import robomimic.utils.obs_utils as ObsUtils
    import robomimic.utils.env_utils as EnvUtils
    from robomimic.envs.env_base import EnvBase
    from robomimic.utils.file_utils import get_env_metadata_from_dataset

    import mimicgen
    import mimicgen.utils.file_utils as MG_FileUtils
    import mimicgen.utils.robomimic_utils as RobomimicUtils
    from mimicgen.utils.misc_utils import add_red_border_to_frame
    from mimicgen.configs import MG_TaskSpec

    assert dataset_dir is not None, "dataset_dir is required"
    assert "image_size" in env_kwargs, "image_size is required"
    self.image_size = env_kwargs["image_size"]
    # assert self.image_size == (84, 84), "We only want to use 84x84 images"
    assert self.image_size == (96, 96), "We only want to use 96x96 images"
    assert task in IMITATION_DATASET_MAPPING, f"Task {task} not found in IMITATION_DATASET_MAPPING"
    dataset_path = str(pathlib.Path(dataset_dir) / IMITATION_DATASET_MAPPING[task])
    self.dataset_path = dataset_path

    # raw_cfg = self._initialize_raw_env()
    self._env = RobosuiteEnv(task, **env_kwargs)
    self.data = h5py.File(dataset_path, "r")
    self.num_demos = len(self.data['data'])

  def _initialize_raw_env(self):
    # need to make sure ObsUtils knows which observations are images, but it doesn't matter
    # for playback since observations are unused. Pass a dummy spec here.
    dummy_spec = dict(
      obs=dict(
        low_dim=["robot0_eef_pos"],
        rgb=[],
        # image=[],
      ),
    )
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)
    env_meta = get_env_metadata_from_dataset(dataset_path=self.dataset_path)
    env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=True)
    return env


  @property
  def env(self) -> Env:
    return self._env

  # def collect(self, final_env: Env, n_episodes: int = 1):
  #   # raise NotImplementedError('Returns: list of list of dicts (episodes -> transitions -> step)')
  #   episode_data = []
  #   for i in range(n_episodes):
  #     id = i % self.num_demos
  #     demo_key = f"demo_{id}"
  #     episode_length = len(self.data['data'][demo_key]['actions'])
  #     transitions = []
  #     for step in range(episode_length):
  #       transition = {}
  #       action = self.data['data'][demo_key]['actions'][step]
  #       obs = self.data['data'][demo_key]['obs']
  #       obs = {k: v[()] for k, v in obs.items() if k in SIMPLE_OBS_KEYS}
  #       transition['action'] = action
  #       transition.update(obs)
  #       transition['reward'] = self.data['data'][demo_key]['rewards'][step]
  #       if step == 0:
  #         transition['is_first'] = True
  #       else:
  #         transition['is_first'] = False
  #       if step == episode_length - 1:
  #         transition['is_last'] = True
  #         transition['is_terminal'] = True
  #         transition['is_success'] = True
  #         transition['has_succeeded'] = True
  #         transition['success_rate'] = 1.0
  #         transition['success'] = True
  #       else:
  #         transition['is_last'] = False
  #         transition['is_terminal'] = False
  #         transition['is_success'] = False
  #         transition['has_succeeded'] = False
  #         transition['success_rate'] = 0.0
  #         transition['success'] = False
  #       transitions.append(transition)
  #     episode_data.append(transitions)
  #   return episode_data

  def dataset(self, final_env: Env):
    i = 0
    while True:
      id = i % self.num_demos
      print(f"[run] [expert] fill_count (ep): {id} / {self.num_demos}", end='\r', color='green')
      demo_key = f"demo_{id}"
      episode_length = len(self.data['data'][demo_key]['actions'])
      for step in range(episode_length):
        transition = {}
        action = self.data['data'][demo_key]['actions'][step]
        obs = self.data['data'][demo_key]['obs']
        obs = {k: np.asarray(v[()][step], dtype=final_env.obs_space[k].dtype) for k, v in obs.items() if k in SIMPLE_OBS_KEYS}
        transition['action'] = np.asarray(final_env.act_space['action'].normalize(action), dtype=np.float32)
        # transition['action'] = action
        transition.update(obs)
        transition['reward'] = np.asarray(self.data['data'][demo_key]['rewards'][step], dtype=np.float32)
        transition['agentview_image'] = np.asarray(cv2.resize(transition['agentview_image'], self.image_size), dtype=np.uint8)
        transition['robot0_eye_in_hand_image'] = np.asarray(cv2.resize(transition['robot0_eye_in_hand_image'], self.image_size), dtype=np.uint8)
        transition['relative_stability'] = 0.0
        if step == 0:
          transition['is_first'] = True
        else:
          transition['is_first'] = False
        if step == episode_length - 1:
          transition['is_last'] = True
          transition['is_terminal'] = True
          transition['is_success'] = True
          transition['has_succeeded'] = True
          transition['success_rate'] = 1.0
          transition['success'] = True
        else:
          transition['is_last'] = False
          transition['is_terminal'] = False
          transition['is_success'] = False
          transition['has_succeeded'] = False
          transition['success_rate'] = 0.0
          transition['success'] = False
        yield transition
      i = (i + 1) % self.num_demos

class RobosuiteEnvMultiTasks(Env):
  # NOTE: Currently, this does not work, lol
  def __init__(self, tasks: List[str], **kwargs):
    self._tasks = tasks
    self._envs = [RobosuiteEnv(task, **kwargs) for task in tasks]

    _spaces: List[Dict[str, Space]] = [c.obs_space for c in self._envs]
    assert all(s == _spaces[0] for s in _spaces), "All tasks must have the same observation space"
    self.obs_space = _spaces[0]

    _spaces = [c.act_space for c in self._envs]
    assert all(s == _spaces[0] for s in _spaces), "All tasks must have the same action space"
    self.act_space = _spaces[0]

    self._current_task_id = np.random.randint(len(tasks))

  @functools.cached_property
  def obs_space(self):
    return {
      **self._envs[0].raw_spaces,
      'reward': Space(np.float32),
      'is_first': Space(bool),
      'is_last': Space(bool),
      'is_terminal': Space(bool),
      'success': Space(bool)
    }

  @functools.cached_property
  def act_space(self):
    low, high = self._envs[0].action_spec
    action = Space(low.dtype, low.shape, low, high)
    return {"reset": Space(bool), "action": action}

  def step(self, action):
    obs = self._envs[self._current_task_id].step(action)
    if obs['is_last']:
      self._current_task_id = np.random.randint(len(self._tasks))
    return obs

  def render(self):
    return self._envs[self._current_task_id].render()
