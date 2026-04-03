#!/bin/bash -l

# Before using this file, install python 3.10, and then activate it

# pytorch CUDA 11.8. install pytorch with conda make jax not realizing gpu
# Torch will use cuda 11.8 while jax will use cuda 12
# I don't think I need pytorch for doing anything
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Jax install CUDA 11 installation
# Note: wheels only available on linux.
# pip install "jax[cuda12]==0.6.1"
# pip install optax==0.2.4 flax==0.10.6
pip install -U jax[cuda12]
pip install optax flax

# install computation lib. Install optax first because it can override jax verions (not what we want)
# This is the 0.24.0 release of TensorFlow Probability. It is tested and stable against TensorFlow 2.16.1 and JAX 0.4.25.
# https://github.com/tensorflow/probability/releases/tag/v0.24.0 => install tf_keras to use keras 2
# pip install tensorflow-cpu==2.19.0 tf-keras==2.19.0 tensorflow-probability==0.25.0
pip install tensorflow-cpu tf-keras tensorflow-probability

# pip install hydra-core==1.3.2 kornia==0.7.2 pandas==2.0.3 wandb==0.17.4 hydra-submitit-launcher==1.2.0 submitit==1.5.2 orjson==3.10.15
# pip install tensordict==0.6.2 torchrl==0.6.0 --no-deps

# Install other realted libraies (gymnasium_robotics requires pettingzoo and imageio). portal for multi processing
pip install portal colored seaborn rich ruamel.yaml==0.17.32 opencv-python opencv-python-headless tensorflow-datasets pandas==2.0.3 scikit-learn scikit-image pybullet pygame # pettingzoo==1.24.3 imageio==2.33.1

# installing rl env libs
# we also need to install the old version of gym since some libs still use that
pip install gymnasium==1.1.1 gym==0.24.1 # robosuite_models==1.0.0 robosuite==1.5.1 # NOTE: We will instead use our version of robosuite, which have some modification for QoL
# dm-control==1.0.14 dm-control
# pip install pygame==2.6.1

# Install box2d dependencies (requires special handling)
# First, install system dependencies for box2d-py
# On Ubuntu/Debian: sudo apt-get install build-essential python3-dev libboost-all-dev
# On conda environments, we can use conda to install these dependencies
conda install -c conda-forge swig boost-cpp -y
# Now install gymnasium with box2d extra (box2d-py should already be installed)
pip install gymnasium[box2d]==1.1.1

# pip install mujoco==3.2.5 # current version of mujoco does not work with robosuite
pip install mujoco==2.3.2 # current version of mujoco does not work with robosuite

# Install stable baseline3 because we need it to do the expert. Training gymnasium robotics requires `TQC` in `sb3-contrib-2.1.0 stable-baselines3-2.1.0`
# this will install pygame matplotlib pandas
# pip install 'stable-baselines3[extra]==2.3.0' 'sb3-contrib==2.3.0'

pip install uvicorn fastapi pydantic

################################## Environments ##################################

# Gym robotics: Franka kitchen
pip install gymnasium-robotics==1.3.1
pip install "minari[all]==0.5.3"
# Setup gymrobot franka kitchen expert dataset using minari
# minari list remote
# minari download D4RL/kitchen/complete-v2
# minari list local

mkdir -p third_party
cd third_party

# Robosuite. NOTE: On this require gcc to build. On RC, we might want to enable
# gcc by `spack load gcc@9.3.0/hufzekv` before doing `bash install.sh`
pip install git+ssh://git@github.com/rxng8/robosuite.git@master
# git clone git@github.com:rxng8/robosuite.git
# cd robosuite
# pip install -e .
# cd ..
# Robosuite requires setting up macro
# This requires python 3.10, otherwise replace the python version in the line
python $CONDA_PREFIX/lib/python3.10/site-packages/robosuite/scripts/setup_macros.py
# Mimicgen
git clone git@github.com:rxng8/mimicgen.git
cd mimicgen
pip install -e .
cd ..
git clone https://github.com/ARISE-Initiative/robomimic.git
cd robomimic
git checkout d0b37cf214bd24fb590d182edb6384333f67b661
pip install cmake==3.31.4 # cmake >=4 cannot build egl_probe, which is required by robomimic
pip install -e .
python robomimic/scripts/setup_macros.py
cd ..
# To test: python mimicgen/mimicgen/scripts/demo_random_action.py
# To download dataset: python download_datasets.py --dataset_type source --tasks all


# Calvin
# pip install setuptools==57.5.0 # pyhash (required by calvin) can only be installed using setuptools <58
# git clone --recurse-submodules git@github.com:rxng8/calvin.git
# export CALVIN_ROOT=$(pwd)/calvin
# cd $CALVIN_ROOT && sh install.sh
# cd .. # to the third_party dir
# Calvin dataset----
# cd $CALVIN_ROOT/dataset
# sh download_data.sh D
# # https://github.com/mees/calvin?tab=readme-ov-file#24-feb-2023
# cd task_D_D
# wget http://calvin.cs.uni-freiburg.de/scene_info_fix/task_D_D_scene_info.zip
# unzip task_D_D_scene_info.zip && rm task_D_D_scene_info.zip
# cd ../..
# -------------------
# setuptools is already updated above after box2d installation

# Language table (our modified version)
pip install git+https://github.com/rxng8/language-table.git


# Finally, to the root of the project
cd ..



####################################################


# Text processing
# we need to install rust for installing transformers
# curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# source $HOME/.cargo/env
pip install tokenizers
pip install transformers

# Install ffmpeg for rendering gifs
conda install -c conda-forge ffmpeg=6.1.1 -y # version 7 does not work


# Some post processing
pip install numpy==1.26.4 # This is the most stable version of numpy
# pip install numpy==2.1.0


pip install 'jax[cuda12]==0.4.33' # new version of jax is buggy


