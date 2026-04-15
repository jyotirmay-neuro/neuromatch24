from pathlib import Path
import sys
import os
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, './tonic')

import tonic.torch

import numpy as np
import collections
import argparse
import os
import yaml
import typing as T
import imageio
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import seaborn as sns
from IPython.display import HTML

import dm_control as dm
import dm_control.suite.swimmer as swimmer
from dm_control.rl import control
from dm_control.utils import rewards
from dm_control import suite
from dm_control.suite.wrappers import pixels

from acme import wrappers

from torch import nn
import tonic

#import swimmer as myswimmer
from tonic.torch import models, normalizers
import torch

from models.model_creators import *

from lxml import etree

_SWIM_SPEED = 0.1

def update_viscocity(model_string, viscocity):
    tree = etree.fromstring(model_string)
    option = tree.find('./option')
    option.set('viscosity', str(viscocity))
    
    return etree.tostring(tree, pretty_print=True)

#@swimmer.SUITE.add()
def swim(
  n_links=6,
  desired_speed=_SWIM_SPEED,
  time_limit=swimmer._DEFAULT_TIME_LIMIT,
  random=None,
  environment_kwargs={},
  global_viscocity=0.0
):
  """Returns the Swim task for a n-link swimmer."""
  #with open('data.txt', 'r') as file:
  #  data = file.read().replace('\n', '')
  model_string, assets = swimmer.get_model_and_assets(n_links)

  model_string = update_viscocity(model_string, global_viscocity) 
  physics = swimmer.Physics.from_xml_string(model_string, assets=assets)
  task = Swim(desired_speed=desired_speed, random=random)
  return control.Environment(
    physics,
    task,
    time_limit=time_limit,
    control_timestep=swimmer._CONTROL_TIMESTEP,
    **environment_kwargs,
  )

@swimmer.SUITE.add()
def swim_vis0(
):
    return swim(global_viscocity=0.0)

@swimmer.SUITE.add()
def swim_vis1(
):
    return swim(global_viscocity=0.1)

@swimmer.SUITE.add()
def swim_vis2(
):
    return swim(global_viscocity=0.2)
    
@swimmer.SUITE.add()
def swim_vis4(
):
    return swim(global_viscocity=0.4)

@swimmer.SUITE.add()
def swim_vis6(
):
    return swim(global_viscocity=0.6)

@swimmer.SUITE.add()
def swim_vis8(
):
    return swim(global_viscocity=0.8)

class Swim(swimmer.Swimmer):
    """Task to swim forwards at the desired speed."""

    def __init__(self, desired_speed=_SWIM_SPEED, **kwargs):
        super().__init__(**kwargs)
        self._desired_speed = desired_speed
        self._target_distance = 2.0

    def initialize_episode(self, physics):
        super().initialize_episode(physics)
        # Hide target by setting alpha to 0.
        physics.named.model.mat_rgba["target", "a"] = 1
        physics.named.model.mat_rgba["target_default", "a"] = 1
        physics.named.model.mat_rgba["target_highlight", "a"] = 1

        # Set the target's position
        target_pos = [0.0, 0.0, 0.02]
        physics.named.model.geom_pos["target"][:3] = target_pos

    def get_observation(self, physics):
        """Returns an observation of joint angles and body velocities."""
        obs = collections.OrderedDict()
        obs["joints"] = physics.joints()
        obs["body_velocities"] = physics.body_velocities()
        return obs

    def get_reward(self, physics):
        """Returns a smooth reward that is 0 when stopped or moving backwards, and rises linearly to 1
        when moving forwards at the desired speed."""
        forward_velocity = -physics.named.data.sensordata["head_vel"][1]
        return rewards.tolerance(
            forward_velocity,
            bounds=(self._desired_speed, float("inf")),
            margin=self._desired_speed,
            value_at_margin=0.0,
            sigmoid="linear",
        )

import contextlib
import io

#with contextlib.redirect_stdout(io.StringIO()): #to suppress output
#    !git clone https://github.com/neuromatch/tonic
#    %cd tonic
    
def play_model(path, checkpoint="last", environment="default", seed=None, header=None):
    """
    Plays a model within an environment and renders the gameplay to a video.

    Parameters:
    - path (str): Path to the directory containing the model and checkpoints.
    - checkpoint (str): Specifies which checkpoint to use ('last', 'first', or a specific ID). 'none' indicates no checkpoint.
    - environment (str): The environment to use. 'default' uses the environment specified in the configuration file.
    - seed (int): Optional seed for reproducibility.
    - header (str): Optional Python code to execute before initializing the model, such as importing libraries.
    """

    if checkpoint == "none":
        # Use no checkpoint, the agent is freshly created.
        checkpoint_path = None
        tonic.logger.log("Not loading any weights")
    else:
        checkpoint_path = os.path.join(path, "checkpoints")
        if not os.path.isdir(checkpoint_path):
            tonic.logger.error(f"{checkpoint_path} is not a directory")
            checkpoint_path = None

        # List all the checkpoints.
        checkpoint_ids = []
        for file in os.listdir(checkpoint_path):
            if file[:5] == "step_":
                checkpoint_id = file.split(".")[0]
                checkpoint_ids.append(int(checkpoint_id[5:]))

        if checkpoint_ids:
            if checkpoint == "last":
                # Use the last checkpoint.
                checkpoint_id = max(checkpoint_ids)
                checkpoint_path = os.path.join(checkpoint_path, f"step_{checkpoint_id}")
            elif checkpoint == "first":
                # Use the first checkpoint.
                checkpoint_id = min(checkpoint_ids)
                checkpoint_path = os.path.join(checkpoint_path, f"step_{checkpoint_id}")
            else:
                # Use the specified checkpoint.
                checkpoint_id = int(checkpoint)
                if checkpoint_id in checkpoint_ids:
                    checkpoint_path = os.path.join(
                        checkpoint_path, f"step_{checkpoint_id}"
                    )
                else:
                    tonic.logger.error(
                        f"Checkpoint {checkpoint_id} not found in {checkpoint_path}"
                    )
                    checkpoint_path = None
        else:
            tonic.logger.error(f"No checkpoint found in {checkpoint_path}")
            checkpoint_path = None

    # Load the experiment configuration.
    arguments_path = os.path.join(path, "config.yaml")
    with open(arguments_path, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    config = argparse.Namespace(**config)

    # Run the header first, e.g. to load an ML framework.
    try:
        if config.header:
            exec(config.header)
        if header:
            exec(header)
    except:
        pass

    # Build the agent.
    agent = eval(config.agent)

    # Build the environment.
    if environment == "default":
        environment = tonic.environments.distribute(lambda: eval(config.environment))
    else:
        environment = tonic.environments.distribute(lambda: eval(environment))
    if seed is not None:
        environment.seed(seed)
    
    print("ENV::", environment.observation_space)

    # Initialize the agent.
    agent.initialize(
        observation_space=environment.observation_space,
        action_space=environment.action_space,
        seed=seed,
    )

    # Load the weights of the agent form a checkpoint.
    print(agent.model)
    if checkpoint_path:
        agent.load(checkpoint_path)

    steps = 0
    test_observations = environment.start()
    #frames = [environment.render("rgb_array", camera_id=0, width=640, height=480)[0]]
    score, length = 0, 0

    while True:
        # Select an action.
        actions = agent.test_step(test_observations, steps)
        assert not np.isnan(actions.sum())

        # Take a step in the environment.
        test_observations, infos = environment.step(actions)
        #frames.append(
        #    environment.render("rgb_array", camera_id=0, width=640, height=480)[0]
        #)
        agent.test_update(**infos, steps=steps)

        score += infos["rewards"][0]
        length += 1

        if infos["resets"][0]:
            break

    video_path = os.path.join(path, "video.mp4")
    print("Reward for the run: ", score)
    return score
    #return display_video(frames, video_path)

def run_multiple(path, num_runs=1, checkpoint="last", environment="default", seed=None, header=None):
    """
    Runs a model multiple times and returns the average score.

    Parameters:
    - path (str): Path to the directory containing the model and checkpoints.
    - checkpoint (str): Specifies which checkpoint to use ('last', 'first', or a specific ID). 'none' indicates no checkpoint.
    - environment (str): The environment to use. 'default' uses the environment specified in the configuration file.
    - seed (int): Optional seed for reproducibility.
    - header (str): Optional Python code to execute before initializing the model, such as importing libraries.
    - num_runs (int): Number of runs to average the score over.

    Returns:
    - float: Average score over the runs.
    """

    scores = []
    for _ in range(num_runs):
        scores.append(play_model(path, checkpoint, environment, seed, header))
    return scores #np.mean(scores)

if __name__ == "__main__":
    # Play the model
    print(sys.argv[1])
    outfile = sys.argv[2]
    time_feature = sys.argv[3]
    envs = ['swim_vis0', 'swim_vis1', 'swim_vis2', 'swim_vis4', 'swim_vis6', 'swim_vis8']
    #viscocities = [0, 1, 2]
    results = []

    for env in envs:
        env_string = f'tonic.environments.ControlSuite(\'swimmer-{env}\', time_feature={time_feature})'
        results.append(
            run_multiple(
                sys.argv[1], 
                10, 
                environment=env_string
            )
        )
    
    out = dict()
    out["envs"] = envs
    out["results"] = torch.tensor(results)
    torch.save(out, outfile)

