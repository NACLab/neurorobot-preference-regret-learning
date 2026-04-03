# Neurorobot Preference Regret Learning (NPRL)

## About the Repository

### Requirements
* Operating System: Linux.
* The system is tested on 1 RTX 3060 and 1 A100.
* [Conda/Miniconda](https://docs.anaconda.com/free/miniconda/) -- python virtual environment/package manager:
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```
* Python: 3.10 (required by ROS 2 if needed)
* Fully tested on Linux 22.04 LTS with 1 Nvidia RTX 3060 GPU

### Structure

* This repository contains a framework that can be used to test different agents on multiple platforms. The framework is designed to be modular, allowing for easy addition of new agents and environments.

* This repository is structured to run experiments on various task, you have to provide an experiment name (through `expname` field) for each run. By default, the experiments are saved in the logs directory, but you can change the log directory by modifying the `logroot` parameter in the configuration files.


* Here is the main structure of the repository:

```
neurorobot-preference-regret-learning/
├── install.sh         # Installation script for dependencies and setup
├── README.md          # Project documentation and usage instructions
│── logs/              # Directory for experiment logs and results
│── data/              # Directory for datasets and collected data
├── lib/               # Core library code
│   ├── agent/         # Agent framework and base classes
│   ├── common/        # Shared utilities and helper modules
│   ├── envs/          # Environment definitions and wrappers (important! Please check this)
│   ├── nlp/           # Natural language processing utilities
│   ├── nn/            # Neural network components and utilities
│   ├── replay/        # Replay buffer and related logic
│   └── utils/         # Additional utility functions
├── nprl/              # Main project package (Neurorobot Preference Regret Learning)
│   ├── agents/        # Actual agent implementations are implemented in here.
│   ├── callbacks.py   # Callback functions to be used during training/evaluation's
│   │                  #   environmental simulations
│   ├── configs.yaml   # Configuration files for the entire pipeline
│   ├── driver.py      # Main driver script (to drive the simulation)
│   ├── networks.py    # Network architectures and modules for QoL (used in agents)
│   └── utils.py       # Utility functions for the framework
├── scripts/           # Experiment and utility scripts
│   ├── collect.py     # Expert data collection script for the experiment
│   ├── eval.py        # Evaluation script
│   ├── export.py      # Given a trained model (usually the expert), export the dataset
│   └── train.py       # Training script
└── third_party/       # External dependencies and third-party
    |                  #   code (automatically pulled by running `bash install.sh`)
    ├── mimicgen/      # MimicGen third-party library
    └── robomimic/     # RoboMimic third-party library

```


### About the CLI and config file

* To train an agent you might just want to run: `python scripts/train.py --configs <primitive1> <primitive2> ... --<key1> <value1> --<key2> <value2> ...` (make sure you check the file to see what it actually does)

* For example, to train the agent on the mountain car environment, you can run:

```bash
python scripts/train.py --logroot logs --expname slampc-mtc-experiment01 --agent slampc --task mtc_mtc --encoder.prockeys 'image|state' --decoder.prockeys 'image|state'  --text_processor '' --.*\.mults 2,3,4,4 --.*\.depth 4 --.*\.units 128 --.*\.deter 512 --.*\.hidden 64 --.*\.classes 4 --batch_size 16 --batch_length 64
```

where `--logroot` specifies the root directory for logs, `--expname` specifies the experiment name, `--agent` specifies the agent type, and `--task` specifies the task/environment. `--encoder.prockeys 'image|state' --decoder.prockeys 'image|state'  --text_processor '' --.*\.mults 2,3,4,4 --.*\.depth 4 --.*\.units 128 --.*\.deter 512 --.*\.hidden 64 --.*\.classes 4` specifies the agent network configuration. Other configurations such as `--batch_size` and `--batch_length` can also be specified.

* You can also run with config primitives to shorten the command line: `python scripts/train.py --configs <primitive1> <primitive2> ...`. For example:

```bash
python scripts/train.py --configs mtc slampc_1m --logroot logs --expname slampc-mtc-experiment01
```
This also works because the config primitive name `mtc` and `slampc_1m` has already been defined in the config file `configs.yaml`

* Structure of the config file. First, the yaml config file contanins multiple key-value pairs, where the first layer of the keys are `defaults` and primitive names. The loader will load the `defaults` first, the based on the provided arguments on the command line you provide, it will merge the config primitives with the defaults (if any). It will also merge the argument you pass in with the defaults to compose a final config (similar to hydra config but a lighter version).

* Note: the order matter: `--slampc_1m --encoder.units 4` will results in the final encoder units being `4` as it is passed later in the command line. The later ones will override the earlier ones if they are the same key.

* Note: the argument you can pass can be searched, implemented, and examined in the config file `configs.yaml`, the `defaults` section. The config primitive can be searched, implemented, and examined in the config file `configs.yaml` for sections other than the `defaults` section.

* We also accept regex argument: `--.*\.foo bar` means that for all keys in all level that matches the regex `.*\.foo`, its value will be set to `bar`. This is useful for setting multiple keys at once. For example, `--.*\.mults 2,3,4,4` will set all keys that match the regex `.*\.mults` to the array value `[2, 3, 4, 4]`.


### Experimental Steps

1. **[Step 1]** Setup, create conda environment, and install dependencies:

```bash
conda create -n nprl pip python=3.10 -y
conda activate nprl
bash install.sh
```

2. **[Step 2]** (Optional) Download the datasets

* If you are running franka kitchen, please setup the franka kitchen expert dataset using minari
```bash
minari list remote
minari download D4RL/kitchen/complete-v2
minari list local
```

* If you are working on other environments, you can complete the step 3 and 4 to collect expert data

* If you want to collect real data (which normally happens when doing real robots), you can skip step 3 and 4. Make sure you process the data and put in the data folder in the following `hdf5` structure (also check the `scripts/export.py` for more information about how we write the structure to `hdf5` file):

```
dataset.h5
keys: episode_000000, episode_000001, ..., episode_abcdef # <---- these are the episode keys, the number is in 6-figure format
values: {
  # All metadata (required)
  'has_succeeded': 'Shape: (n,). Dtype: bool', # whether the agent has succeeded in this episode until this step
  'is_first': 'Shape: (n,). Dtype: bool', # whether this is the first step in the trajectory
  'is_last': 'Shape: (n,). Dtype: bool', # whether this is the last step in the trajectory (can be either due to agent's action or environment's truncation)
  'is_success': 'Shape: (n,). Dtype: bool', # whether the agent success at a state
  'is_terminal': 'Shape: (n,). Dtype: bool', # whether the episode terminate on agent's action (not environment truncation)
  'relative_stability': 'Shape: (n,). Dtype: float32', # you can set this to any number, that is fine
  'reward': 'Shape: (n,). Dtype: float32',
  'success_rate': 'Shape: (n,). Dtype: float32' # success rate of the last 100 episodes

  # All observations and actions (not necessary be the following, but recommended), for example:
  'action': 'Shape: (n, action_dim). Dtype: float32',
  'image': 'Shape: (n, 64, 64, 3). Dtype: uint8',
  'state': 'Shape: (n, 2). Dtype: float32',
}
```

* note that the `n` in there is the length of the trajectory

* Then in your config file, you will have to specify the dataset path in the `expert.<env-suite>` field, for example:

```
expert:
  mtc: { dataset_path: data/mtc/dataset.h5 }
```
Or you can specify it in the command line (which is what we will do in the next step)


3. **[Step 3]** Train the expert model.

* Here, we will train an agent on the mountain car environment. You can also train other agents on other environments, but make sure you have the dataset ready.

```bash
python scripts/train.py --configs mtc slampc_1m denserew --agent slampc --expname slampc-mtc-expert --run.steps 3000000
```

* Note that the `denserew` config primitive specifies that the environment will use dense reward instead of sparse reward (sparse reward is the default)
* The trained agent's weights are saved inside the experiment folder. The agent automatically loads the weights when you run any scripts that call `--expname` to that folder.


1. **[Step 4]** Use the trained expert, simulate it again on the environment to export the expert dataset (will be stored in `data/mtc/` directory):

```bash
python scripts/export.py logs/slampc-mtc-expert data/mtc 5
```

* Note that the export.py argument is different, it comes in the form: `python scripts/export.py <logdir-of-trained-expert> <target-data-file> <number-of-episodes-to-export>`

* Note that the agent automatically loads the config file in the experiment folder `logdir` and loads the weights inside that experiment folder.

* The target dataset file is a `hdf5` file and will have the path `data/mtc/dataset.h5`.


5. **[Step 5]** Collect expert demonstrations:

* This part is where you run the actual experiment. You will have your agent consume the data (that was collected before). By consumption, we mean, the agent save the data to its replay buffer (or memories) togther with its inferred state (if any). This step is required if we are training the agent that leverage expert demonstrations.

```bash
python scripts/collect.py --configs mtc slampc_1m --agent slampc --expname slampc-mtc --expert.mtc.dataset_path data/mtc/dataset.h5
```

* You can refer to the `mtc` primitive in `configs.yaml` to see all setup


6. **[Step 6]** Run the actual agent:

```bash
python scripts/train.py --configs mtc slampc_1m --agent slampc --expname slampc-mtc
```

7. **[Step 7]** Observe the results on tensorboard: `tensorboard --logdir logs`

* If you want to run tensorboard on a specific port: `tensorboard --logdir logs --port <your_port_number>`


8. **[Step 8]** Evaluate the agent:

```bash
python scripts/eval.py --configs mtc slampc_1m --agent slampc --expname slampc-mtc
```


### Writing Your Own Environments

* If you want to write your own environments, you can do so by creating a new file in the `lib/envs/` directory. The environment should inherit from the `Env` or `ActiveEnv` class (`lib/envs/base.py`) and implement the required methods. For reference, please check the existing environments in the `lib/envs/` directory.

#### 1. Philosophy

* `suite` and `task`: In this framework, we use the term `suite` to refer to a collection of tasks. Generally, a suite is implemented under a single environment class, and the task string is passed in its argument to determine the logic of the environment. For example, in the `lib/envs/metaworld.py`, we have a `MetaWorldEnv` class that implements the Meta-World suite, and the task is passed in as an argument to the constructor. The task string is used to determine which task to run in the environment. `Meta-World` itself is a suite/collection of a lot of tasks. By default, you would provide in the config file the task string: `<suite>_<task>`. For example, `metaworld_button-press-v2` is a task in the Meta-World suite. The `metaworld` is the suite name, and `button-press-v2` is the task name. The task string is used to determine which task to run in the environment. Note that the task string will always be passed in as the first argument to the environment constructor. For more information, check the `lib/envs/build.py` file, which is used to build the environment.

* For every environment, it is required to have the `obs_space` and `act_space` property. `obs_space` denotes the observation space of the environment, and `act_space` denotes the action space of the environment. `obs_space` and `act_space` has to be a dictionary, and the actual observation and action has to be dictionaries as well.

* The observation space must have the following:
  * `is_first`: a boolean indicating whether this is the first step in the episode.
  * `is_last`: a boolean indicating whether this is the last step in the episode (it can both be due to the agent's action that terminates the episode or the environment's truncation).
  * `is_terminal`: a boolean indicating whether the episode terminates on the agent's action (not environment truncation).
  * `reward`: a float indicating the reward received by the agent at this step.

* The action space must have the following:
  * `reset`: a boolean indicating whether the action is a reset action. This is used to reset the environment when the agent decides to reset it.

* An environment must have the following methods:
  * `is_success(self, obs: Dict[str, np.ndarray]) -> bool` (Optional): This will be used by the wrapper (`lib/envs/wrappers.py`) to determine whether the agent has succeeded in the episode. If not implemented, the wrapper will assume that the agent has not succeeded.
  * `step(self, action: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]`: This method takes an action dictionary and returns an observation dictionary. The reset of environment is also handled here in this method (either by the agent's action's `reset` key or the environment's internal logic).
  * `close(self)`: This method is called when the environment is closed. It should be used to clean up any resources used by the environment. You can leave it empty if there is no resource to clean up.
  * `render(self) -> Dict[str, np.ndarray] | np.ndarray | List[np.ndarray]`: This method is called to render the environment. It should return a dictionary or list of rendered images or a single image. Generally, this render method is called in `scripts/train.py` for loggin purposes.


#### 2. Implementing an Actual Environment

* We will be using the Meta-World environment as an example to implement a new environment.

##### 2.1. Create an environment

* First we need to inherit from the `Env` class in `lib/env`
* Then we need to implement the `obs_space` and `act_space` properties. The `obs_space` should contain the observation space of the environment, and the `act_space` should contain the action space of the environment. In this case, the action space has the must-have `reset` key, and then the `action` key which is the actual action of the Meta-World robot. For observation, Meta-World returns a dictionary of observations, so we flatten dictionary to avoid any nested dictionary issues (see `self._flatten` and `self._unflatten`). We also include what every the information in the information dictionary (normally returned by the gym environment) just in case we need it later.

* Next, we implement the `step` logic, which handle the actual reset (if needed) and the step logic of the environment. This will return a dictionary of observations, which must include the `is_first`, `is_last`, `is_terminal`, and `reward` keys in the observation space `obs_space`. It also must include other observation keys mentioned in the `obs_space` propert as well.

* After that, we finish implementing other methods such as `is_success`, `close`, and `render`.

##### 2.2. Integrate the environment

* After implementing the environment, we need to integrate it into the framework. This is done by adding the path to the environment to the `build_env` method in `lib/envs/build.py`.
* In the `ctor` dictionary, we need to add a key-value pair with:
  * key: any non-spaced name (that represents a suite) that you want to use to refer to the environment (this will be used as main reference point in the config file `myoe/configs.yaml`).
  * value: the full path to the environment class (e.g. `lib.envs.metaworld.metaworld:MetaWorldEnv`).

* Finally, you have to edit the config file `myoe/configs.yaml` in the `defaults -> env` section if you want to add the config/parameters for the environment. For example:

```yaml
defaults:
  ... # Some other configs here
  env:
    # metaworld is the suite name (the key in the ctor dictionary)
    # followed by the config dict (you can pass any config)
    metaworld: {image_size: [64, 64], reward_shaping: False, action_repeat: 2}
  ... # Other env configs here
... # Other configs here
```

After that, you can define any primitives for environment that you like, for example:

```yaml
defaults:
  ... # Some configs here

... # Some other primitive configs here

# Note: for the primitive configs: for any config, you pass, make sure the key also exists in the defaults section, otherwise it will cause an error
metaworld:
  encoder.prockeys: '.*image.*|.*state.*|obj_to_target|near_object' # This regex tells the encoder to get all the relevant keys in the observation dictionary. In this case: the rgb image, the robot state, the object to target vector, and the near object vector.
  decoder.prockeys: '.*image.*|.*state.*|obj_to_target|near_object' # This regex tells the decoder to decoder all its predictions to the relevant keys in the observation dictionary. In this case: the rgb image, the robot state, the object to target vector, and the near object vector.
  query_encoder.prockeys: '^$' # This regex tells the query encoder to not encode any query, which is fine for this environment. Query encoder is often used when there is a goal provided in the observation dictionary. For example, in the kitchen environment, the query encoder is used to encode the goal state and the natual language instruction of the robot
  env.metaworld: { image_size: [64, 64], reward_shaping: False }
  .*\.mults: [2, 3, 4, 4]

  # The following is the expert configs (if collected using step 4)
  # If you don't have the expert dataset, you can skip this part
  expert: # This is the expert dataset that we collected in step 4
    enable: True # Enable the expert dataset
    n_episodes: 5 # Number of episodes to collect from the expert dataset
    metaworld: { dataset_path: tobeset } # We will set this in the command line later

  # This is for use of changing task in the environment. Default to False for now
  multi_task: False

  # By default, the task will also be passed in the env class as the task string (first parameter in the constructor) as mentioned above
  task: "metaworld_button-press-v2"

... # Some other primitive configs here

```

* Now you can run the agent in your environment by following the steps in the Experimental Steps section above. For example, you can run:

```bash
python scripts/train.py --configs metaworld slampc_1m --agent slampc --expname slampc-metaworld
```

