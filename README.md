# cse574
Project 574
OpenAI Gym
**********
Commands to run and render the environments :

```
python run_mujoco.py --saves --wsaves --opt 2 --env Walker2d-v2 --seed 777 --app savename --dc 0.1
python run_mujoco.py --saves --wsaves --opt 2 --env Hopper-v2 --seed 777 --app savename --dc 0.1
python run_mujoco.py --saves --wsaves --opt 2 --env HalfCheetah-v2 --seed 777 --app savename --dc 0.1

```
Requirements:

Python3
Tensorflow
OpenAI Gym
baselines
mujoco-py



Instructions for instaling OpenAI Gym
=====================================

**OpenAI Gym is a toolkit for developing and comparing reinforcement learning algorithms.** This is the ``gym`` open-source library, which gives you access to a standardized set of environments.

.. image:: https://travis-ci.org/openai/gym.svg?branch=master
    :target: https://travis-ci.org/openai/gym

`See What's New section below <#what-s-new>`_

``gym`` makes no assumptions about the structure of your agent, and is compatible with any numerical computation library, such as TensorFlow or Theano. You can use it from Python code, and soon from other languages.

If you're not sure where to start, we recommend beginning with the
`docs <https://gym.openai.com/docs>`_ on our site. See also the `FAQ <https://github.com/openai/gym/wiki/FAQ>`_.

A whitepaper for OpenAI Gym is available at http://arxiv.org/abs/1606.01540, and here's a BibTeX entry that you can use to cite it in a publication::


.. contents:: **Contents of this document**
   :depth: 2

Basics
======

There are two basic concepts in reinforcement learning: the
environment (namely, the outside world) and the agent (namely, the
algorithm you are writing). The agent sends `actions` to the
environment, and the environment replies with `observations` and
`rewards` (that is, a score).

The core `gym` interface is `Env <https://github.com/openai/gym/blob/master/gym/core.py>`_, which is
the unified environment interface. There is no interface for agents;
that part is left to you. The following are the ``Env`` methods you
should know:

- `reset(self)`: Reset the environment's state. Returns `observation`.
- `step(self, action)`: Step the environment by one timestep. Returns `observation`, `reward`, `done`, `info`.
- `render(self, mode='human', close=False)`: Render one frame of the environment. The default mode will do something human friendly, such as pop up a window. Passing the `close` flag signals the renderer to close any such windows.

Installation
============

You can perform a minimal install of ``gym`` with:

.. code:: shell

    git clone https://github.com/openai/gym.git
    cd gym
    pip install -e .

If you prefer, you can do a minimal install of the packaged version directly from PyPI:

.. code:: shell

    pip install gym

You'll be able to run a few environments right away:

- algorithmic
- toy_text
- classic_control (you'll need ``pyglet`` to render though)

We recommend playing with those environments at first, and then later
installing the dependencies for the remaining environments.

Installing everything
---------------------

To install the full set of environments, you'll need to have some system
packages installed. We'll build out the list here over time; please let us know
what you end up installing on your platform. Also, take a look at the docker files (test.dockerfile.xx.xx) to 
see the composition of our CI-tested images. 

On OSX:

.. code:: shell

    brew install cmake boost boost-python sdl2 swig wget

On Ubuntu 14.04 (non-mujoco only):

.. code:: shell

    apt-get install libjpeg-dev cmake swig python-pyglet python3-opengl libboost-all-dev \
            libsdl2-2.0.0 libsdl2-dev libglu1-mesa libglu1-mesa-dev libgles2-mesa-dev \
            freeglut3 xvfb libav-tools


On Ubuntu 16.04:

.. code:: shell

    apt-get install -y python-pyglet python3-opengl zlib1g-dev libjpeg-dev patchelf \
            cmake swig libboost-all-dev libsdl2-dev libosmesa6-dev xvfb ffmpeg

On Ubuntu 18.04:

.. code:: shell

    apt install -y python3-dev zlib1g-dev libjpeg-dev cmake swig python-pyglet python3-opengl libboost-all-dev libsdl2-dev \
        libosmesa6-dev patchelf ffmpeg xvfb


MuJoCo has a proprietary dependency we can't set up for you. Follow
the
`instructions <https://github.com/openai/mujoco-py#obtaining-the-binaries-and-license-key>`_
in the ``mujoco-py`` package for help.

Once you're ready to install everything, run ``pip install -e '.[all]'`` (or ``pip install 'gym[all]'``).

Supported systems
-----------------

We currently support Linux and OS X running Python 2.7 or 3.5. Some users on OSX + Python3 may need to run

.. code:: shell

    brew install boost-python --with-python3

If you want to access Gym from languages other than python, we have limited support for non-python
frameworks, such as lua/Torch, using the OpenAI Gym `HTTP API <https://github.com/openai/gym-http-api>`_.

Pip version
-----------

To run ``pip install -e '.[all]'``, you'll need a semi-recent pip.
Please make sure your pip is at least at version ``1.5.0``. You can
upgrade using the following: ``pip install --ignore-installed
pip``. Alternatively, you can open `setup.py
<https://github.com/openai/gym/blob/master/setup.py>`_ and
install the dependencies by hand.

Rendering on a server
---------------------

If you're trying to render video on a server, you'll need to connect a
fake display. The easiest way to do this is by running under
``xvfb-run`` (on Ubuntu, install the ``xvfb`` package):

.. code:: shell

     xvfb-run -s "-screen 0 1400x900x24" bash

Installing dependencies for specific environments
-------------------------------------------------

If you'd like to install the dependencies for only specific
environments, see `setup.py
<https://github.com/openai/gym/blob/master/setup.py>`_. We
maintain the lists of dependencies on a per-environment group basis.

Environments
============

The code for each environment group is housed in its own subdirectory
`gym/envs
<https://github.com/openai/gym/blob/master/gym/envs>`_. The
specification of each task is in `gym/envs/__init__.py
<https://github.com/openai/gym/blob/master/gym/envs/__init__.py>`_. It's
worth browsing through both.

Algorithmic
-----------

These are a variety of algorithmic tasks, such as learning to copy a
sequence.

.. code:: python

    import gym
    env = gym.make('Copy-v0')
    env.reset()
    env.render()

Atari
-----

The Atari environments are a variety of Atari video games. If you didn't do the full install, you can install dependencies via ``pip install -e '.[atari]'`` (you'll need ``cmake`` installed) and then get started as follow:

.. code:: python

    import gym
    env = gym.make('SpaceInvaders-v0')
    env.reset()
    env.render()

This will install ``atari-py``, which automatically compiles the `Arcade Learning Environment <http://www.arcadelearningenvironment.org/>`_. This can take quite a while (a few minutes on a decent laptop), so just be prepared.

Box2d
-----------

Box2d is a 2D physics engine. You can install it via  ``pip install -e '.[box2d]'`` and then get started as follow:

.. code:: python

    import gym
    env = gym.make('LunarLander-v2')
    env.reset()
    env.render()

Classic control
---------------

These are a variety of classic control tasks, which would appear in a typical reinforcement learning textbook. If you didn't do the full install, you will need to run ``pip install -e '.[classic_control]'`` to enable rendering. You can get started with them via:

.. code:: python

    import gym
    env = gym.make('CartPole-v0')
    env.reset()
    env.render()

MuJoCo
------

`MuJoCo <http://www.mujoco.org/>`_ is a physics engine which can do
very detailed efficient simulations with contacts. It's not
open-source, so you'll have to follow the instructions in `mujoco-py
<https://github.com/openai/mujoco-py#obtaining-the-binaries-and-license-key>`_
to set it up. You'll have to also run ``pip install -e '.[mujoco]'`` if you didn't do the full install.

.. code:: python

    import gym
    env = gym.make('Humanoid-v2')
    env.reset()
    env.render()


Robotics
------

`MuJoCo <http://www.mujoco.org/>`_ is a physics engine which can do
very detailed efficient simulations with contacts and we use it for all robotics environments. It's not
open-source, so you'll have to follow the instructions in `mujoco-py
<https://github.com/openai/mujoco-py#obtaining-the-binaries-and-license-key>`_
to set it up. You'll have to also run ``pip install -e '.[robotics]'`` if you didn't do the full install.

.. code:: python

    import gym
    env = gym.make('HandManipulateBlock-v0')
    env.reset()
    env.render()

You can also find additional details in the accompanying `technical report <https://arxiv.org/abs/1802.09464>`_ and `blog post <https://blog.openai.com/ingredients-for-robotics-research/>`_.
If you use these environments, you can cite them as follows::

  @misc{1802.09464,
    Author = {Matthias Plappert and Marcin Andrychowicz and Alex Ray and Bob McGrew and Bowen Baker and Glenn Powell and Jonas Schneider and Josh Tobin and Maciek Chociej and Peter Welinder and Vikash Kumar and Wojciech Zaremba},
    Title = {Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research},
    Year = {2018},
    Eprint = {arXiv:1802.09464},
  }

Toy text
--------

Toy environments which are text-based. There's no extra dependency to install, so to get started, you can just do:

.. code:: python

    import gym
    env = gym.make('FrozenLake-v0')
    env.reset()
    env.render()

Examples
========

See the ``examples`` directory.

- Run `examples/agents/random_agent.py <https://github.com/openai/gym/blob/master/examples/agents/random_agent.py>`_ to run an simple random agent.
- Run `examples/agents/cem.py <https://github.com/openai/gym/blob/master/examples/agents/cem.py>`_ to run an actual learning agent (using the cross-entropy method).
- Run `examples/scripts/list_envs <https://github.com/openai/gym/blob/master/examples/scripts/list_envs>`_ to generate a list of all environments.

Testing
=======

We are using `pytest <http://doc.pytest.org>`_ for tests. You can run them via:

.. code:: shell

    pytest

Instructions to install Baselines:
==================================

<img src="data/logo.jpg" width=25% align="right" /> [![Build status](https://travis-ci.org/openai/baselines.svg?branch=master)](https://travis-ci.org/openai/baselines)

# Baselines

OpenAI Baselines is a set of high-quality implementations of reinforcement learning algorithms.

These algorithms will make it easier for the research community to replicate, refine, and identify new ideas, and will create good baselines to build research on top of. Our DQN implementation and its variants are roughly on par with the scores in published papers. We expect they will be used as a base around which new ideas can be added, and as a tool for comparing a new approach against existing ones. 

## Prerequisites 
Baselines requires python3 (>=3.5) with the development headers. You'll also need system packages CMake, OpenMPI and zlib. Those can be installed as follows
### Ubuntu 
    
```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```
    
### Mac OS X
Installation of system packages on Mac requires [Homebrew](https://brew.sh). With Homebrew installed, run the following:
```bash
brew install cmake openmpi
```
    
## Virtual environment
From the general python package sanity perspective, it is a good idea to use virtual environments (virtualenvs) to make sure packages from different projects do not interfere with each other. You can install virtualenv (which is itself a pip package) via
```bash
pip install virtualenv
```
Virtualenvs are essentially folders that have copies of python executable and all python packages.
To create a virtualenv called venv with python3, one runs 
```bash
virtualenv /path/to/venv --python=python3
```
To activate a virtualenv: 
```
. /path/to/venv/bin/activate
```
More thorough tutorial on virtualenvs and options can be found [here](https://virtualenv.pypa.io/en/stable/) 


## Installation
- Clone the repo and cd into it:
    ```bash
    git clone https://github.com/openai/baselines.git
    cd baselines
    ```
- If you don't have TensorFlow installed already, install your favourite flavor of TensorFlow. In most cases, 
    ```bash 
    pip install tensorflow-gpu # if you have a CUDA-compatible gpu and proper drivers
    ```
    or 
    ```bash
    pip install tensorflow
    ```
    should be sufficient. Refer to [TensorFlow installation guide](https://www.tensorflow.org/install/)
    for more details. 

- Install baselines package
    ```bash
    pip install -e .
    ```

### MuJoCo
Some of the baselines examples use [MuJoCo](http://www.mujoco.org) (multi-joint dynamics in contact) physics simulator, which is proprietary and requires binaries and a license (temporary 30-day license can be obtained from [www.mujoco.org](http://www.mujoco.org)). Instructions on setting up MuJoCo can be found [here](https://github.com/openai/mujoco-py)

## Testing the installation
All unit tests in baselines can be run using pytest runner:
```
pip install pytest
pytest
```

## Training models
Most of the algorithms in baselines repo are used as follows:
```bash
python -m baselines.run --alg=<name of the algorithm> --env=<environment_id> [additional arguments]
```
### Example 1. PPO with MuJoCo Humanoid
For instance, to train a fully-connected network controlling MuJoCo humanoid using PPO2 for 20M timesteps
```bash
python -m baselines.run --alg=ppo2 --env=Humanoid-v2 --network=mlp --num_timesteps=2e7
```
Note that for mujoco environments fully-connected network is default, so we can omit `--network=mlp`
The hyperparameters for both network and the learning algorithm can be controlled via the command line, for instance:
```bash
python -m baselines.run --alg=ppo2 --env=Humanoid-v2 --network=mlp --num_timesteps=2e7 --ent_coef=0.1 --num_hidden=32 --num_layers=3 --value_network=copy
```
will set entropy coefficient to 0.1, and construct fully connected network with 3 layers with 32 hidden units in each, and create a separate network for value function estimation (so that its parameters are not shared with the policy network, but the structure is the same)

See docstrings in [common/models.py](baselines/common/models.py) for description of network parameters for each type of model, and 
docstring for [baselines/ppo2/ppo2.py/learn()](baselines/ppo2/ppo2.py#L152) for the description of the ppo2 hyperparamters. 

### Example 2. DQN on Atari 
DQN with Atari is at this point a classics of benchmarks. To run the baselines implementation of DQN on Atari Pong:
```
python -m baselines.run --alg=deepq --env=PongNoFrameskip-v4 --num_timesteps=1e6
```

## Saving, loading and visualizing models
The algorithms serialization API is not properly unified yet; however, there is a simple method to save / restore trained models. 
`--save_path` and `--load_path` command-line option loads the tensorflow state from a given path before training, and saves it after the training, respectively. 
Let's imagine you'd like to train ppo2 on Atari Pong,  save the model and then later visualize what has it learnt.
```bash
python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=2e7 --save_path=~/models/pong_20M_ppo2
```
This should get to the mean reward per episode about 20. To load and visualize the model, we'll do the following - load the model, train it for 0 steps, and then visualize: 
```bash
python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=0 --load_path=~/models/pong_20M_ppo2 --play
```

*NOTE:* At the moment Mujoco training uses VecNormalize wrapper for the environment which is not being saved correctly; so loading the models trained on Mujoco will not work well if the environment is recreated. If necessary, you can work around that by replacing RunningMeanStd by TfRunningMeanStd in [baselines/common/vec_env/vec_normalize.py](baselines/common/vec_env/vec_normalize.py#L12). This way, mean and std of environment normalizing wrapper will be saved in tensorflow variables and included in the model file; however, training is slower that way - hence not including it by default


## Using baselines with TensorBoard
Baselines logger can save data in the TensorBoard format. To do so, set environment variables `OPENAI_LOG_FORMAT` and `OPENAI_LOGDIR`:
```bash
export OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' # formats are comma-separated, but for tensorboard you only really need the last one
export OPENAI_LOGDIR=path/to/tensorboard/data
```
And you can now start TensorBoard with:
```bash
tensorboard --logdir=$OPENAI_LOGDIR
```
## Subpackages

- [A2C](baselines/a2c)
- [ACER](baselines/acer)
- [ACKTR](baselines/acktr)
- [DDPG](baselines/ddpg)
- [DQN](baselines/deepq)
- [GAIL](baselines/gail)
- [HER](baselines/her)
- [PPO1](baselines/ppo1) (obsolete version, left here temporarily)
- [PPO2](baselines/ppo2) 
- [TRPO](baselines/trpo_mpi)



## Benchmarks
Results of benchmarks on Mujoco (1M timesteps) and Atari (10M timesteps) are available 
[here for Mujoco](https://htmlpreview.github.com/?https://github.com/openai/baselines/blob/master/benchmarks_mujoco1M.htm) 
and
[here for Atari](https://htmlpreview.github.com/?https://github.com/openai/baselines/blob/master/benchmarks_atari10M.htm) 
respectively. Note that these results may be not on the latest version of the code, particular commit hash with which results were obtained is specified on the benchmarks page. 

Instructions to install mujoco-py:
==================================

# mujoco-py [![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://openai.github.io/mujoco-py/build/html/index.html) [![Build Status](https://travis-ci.org/openai/mujoco-py.svg?branch=master)](https://travis-ci.org/openai/mujoco-py) [![Build status](https://ci.appveyor.com/api/projects/status/iw52c0198j87s76w?svg=true)](https://ci.appveyor.com/project/wojzaremba/mujoco-py)

[MuJoCo](http://mujoco.org/) is a physics engine for detailed, efficient rigid body simulations with contacts. `mujoco-py` allows using MuJoCo from Python 3.

## Synopsis

### Requirements

The following platforms are currently supported:

- Linux with Python 3.6. See [the `Dockerfile`](Dockerfile) for the canonical list of system dependencies.
- OS X with Python 3.6.
- Windows (experimental) with Python 3.6. See [the Appveyor file](https://github.com/openai/mujoco-py/blob/master/.appveyor.yml#L16-L32) for the canonical list of dependencies.

Python 2 has been desupported since [1.50.1.0](https://github.com/openai/mujoco-py/releases/tag/1.50.1.0). Python 2 users can stay on the [`0.5` branch](https://github.com/openai/mujoco-py/tree/0.5). The latest release there is [`0.5.7`](https://github.com/openai/mujoco-py/releases/tag/0.5.7) which can be installed with `pip install mujoco-py==0.5.7`.

### Install MuJoCo

1. Obtain a 30-day free trial on the [MuJoCo website](https://www.roboti.us/license.html)
   or free license if you are a student.
   The license key will arrive in an email with your username and password.
2. Download the MuJoCo version 1.50 binaries for
   [Linux](https://www.roboti.us/download/mjpro150_linux.zip),
   [OSX](https://www.roboti.us/download/mjpro150_osx.zip), or
   [Windows](https://www.roboti.us/download/mjpro150_win64.zip).
3. Unzip the downloaded `mjpro150` directory into `~/.mujoco/mjpro150`,
   and place your license key (the `mjkey.txt` file from your email)
   at `~/.mujoco/mjkey.txt`.

If you want to specify a nonstandard location for the key and package,
use the env variables `MUJOCO_PY_MJKEY_PATH` and `MUJOCO_PY_MJPRO_PATH`.

### Install and use `mujoco-py`
To include `mujoco-py` in your own package, add it to your requirements like so:
```
mujoco-py<1.50.2,>=1.50.1
```
To play with `mujoco-py` interactively, follow these steps:
```
$ pip3 install -U 'mujoco-py<1.50.2,>=1.50.1'
$ python3
import mujoco_py
from os.path import dirname
model = mujoco_py.load_model_from_path(dirname(dirname(mujoco_py.__file__))  +"/xmls/claw.xml")
sim = mujoco_py.MjSim(model)

print(sim.data.qpos)
# [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]

sim.step()
print(sim.data.qpos)
# [  2.09217903e-06  -1.82329050e-12  -1.16711384e-07  -4.69613872e-11
#   -1.43931860e-05   4.73350204e-10  -3.23749942e-05  -1.19854057e-13
#   -2.39251380e-08  -4.46750545e-07   1.78771599e-09  -1.04232280e-08]
```

See the [full documentation](https://openai.github.io/mujoco-py/build/html/index.html) for advanced usage.

### Missing GLFW

A common error when installing is:

    raise ImportError("Failed to load GLFW3 shared library.")

Which happens when the `glfw` python package fails to find a GLFW dynamic library.

MuJoCo ships with its own copy of this library, which can be used during installation.

Add the path to the mujoco bin directory to your dynamic loader:

    LD_LIBRARY_PATH=$HOME/.mujoco/mjpro150/bin pip install mujoco-py

This is particularly useful on Ubuntu 14.04, which does not have a GLFW package.

## Usage Examples

A number of examples demonstrating some advanced features of `mujoco-py` can be found in [`examples/`](/./examples/). These include:
- [`body_interaction.py`](./examples/body_interaction.py): shows interactions between colliding bodies
- [`disco_fetch.py`](./examples/disco_fetch.py): shows how `TextureModder` can be used to randomize object textures
- [`internal_functions.py`](./examples/internal_functions.py): shows how to call raw mujoco functions like `mjv_room2model`
- [`markers_demo.py`](./examples/markers_demo.py): shows how to add visualization-only geoms to the viewer
- [`serialize_model.py`](./examples/serialize_model.py): shows how to save and restore a model
- [`setting_state.py`](./examples/setting_state.py):  shows how to reset the simulation to a given state
- [`tosser.py`](./examples/tosser.py): shows a simple actuated object sorting robot application

See the [full documentation](https://openai.github.io/mujoco-py/build/html/index.html) for advanced usage.

## Development

To run the provided unit and integrations tests:

```
make test
```

To test GPU-backed rendering, run:

```
make test_gpu
```

This is somewhat dependent on internal OpenAI infrastructure at the moment, but it should run if you change the `Makefile` parameters for your own setup.

## Changelog

- 03/08/2018: We removed MjSimPool, because most of benefit one can get with multiple processes having single simulation.

## Credits

`mujoco-py` is maintained by the OpenAI Robotics team. Contributors include:

- Alex Ray
- Bob McGrew
- Jonas Schneider
- Jonathan Ho
- Peter Welinder
- Wojciech Zaremba
