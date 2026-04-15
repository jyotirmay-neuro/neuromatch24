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

from utils import write_video

from models.model_creators import *

from tasks.swim_in_viscous_spheres import load_env

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
    frames = [environment.render("rgb_array", camera_id=0, width=640, height=480)[0]]
    score, length = 0, 0

    while True:
        # Select an action.
        actions = agent.test_step(test_observations, steps)
        assert not np.isnan(actions.sum())

        # Take a step in the environment.
        test_observations, infos = environment.step(actions)
        frames.append(
            environment.render("rgb_array", camera_id=0, width=640, height=480)[0]
        )
        agent.test_update(**infos, steps=steps)

        score += infos["rewards"][0]
        length += 1

        if infos["resets"][0]:
            break

    video_path = os.path.join(path, "video.mp4")
    print("Reward for the run: ", score)
    print(video_path)
    write_video(video_path, frames)
    return score

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
    envs = ['swim_in_viscous_spheres']
    #viscocities = [0, 1, 2]
    results = []

    for env in envs:
        env_string = f'tonic.environments.ControlSuite(\'swimmer-{env}\', time_feature={time_feature})'
        results.append(
            run_multiple(
                sys.argv[1], 
                1, 
                environment=env_string
            )
        )
    
    out = dict()
    out["envs"] = envs
    out["results"] = torch.tensor(results)
    torch.save(out, outfile)

