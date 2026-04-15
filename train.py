
from pathlib import Path
import sys
import os
sys.path.insert(1, './tonic')
import tonic.torch
import tonic

import dm_control as dm
import dm_control.suite.swimmer as swimmer
from dm_control.rl import control
from dm_control.utils import rewards
from dm_control import suite
from dm_control.suite.wrappers import pixels

from acme import wrappers

from models.model_creators import *

def train(
    header,
    agent,
    environment,
    name="test",
    trainer="tonic.Trainer()",
    before_training=None,
    after_training=None,
    parallel=1,
    sequential=1,
    seed=0,
):
    """
    Some additional parameters:

    - before_training: Python code to execute immediately before the training loop commences, suitable for setup actions needed after initialization but prior to training.
    - after_training: Python code to run once the training loop concludes, ideal for teardown or analytical purposes.
    - parallel: The count of environments to execute in parallel. Limited to 1 in a Colab notebook, but if additional resources are available, this number can be increased to expedite training.
    - sequential: The number of sequential steps the environment runs before sending observations back to the agent. This setting is useful for temporal batching. It can be disregarded for this tutorial's purposes.
    - seed: The experiment's random seed, guaranteeing the reproducibility of the training process.

    """
    # Capture the arguments to save them, e.g. to play with the trained agent.
    args = dict(locals())

    # Run the header first, e.g. to load an ML framework.
    if header:
        exec(header)

    # Build the train and test environments.
    _environment = environment
    environment = tonic.environments.distribute(
        lambda: eval(_environment), parallel, sequential
    )
    test_environment = tonic.environments.distribute(lambda: eval(_environment))

    # Build the agent.
    agent = eval(agent)
    agent.initialize(
        observation_space=test_environment.observation_space,
        action_space=test_environment.action_space,
        seed=seed,
    )

    # Choose a name for the experiment.
    if hasattr(test_environment, "name"):
        environment_name = test_environment.name
    else:
        environment_name = test_environment.__class__.__name__
    if not name:
        if hasattr(agent, "name"):
            name = agent.name
        else:
            name = agent.__class__.__name__
        if parallel != 1 or sequential != 1:
            name += f"-{parallel}x{sequential}"

    # Initialize the logger to save data to the path environment/name/seed.
    path = os.path.join("data", "local", "experiments", "tonic", environment_name, name)
    tonic.logger.initialize(path, script_path=None, config=args)

    # Build the trainer.
    trainer = eval(trainer)
    trainer.initialize(
        agent=agent,
        environment=environment,
        test_environment=test_environment,
    )
    # Run some code before training.
    if before_training:
        exec(before_training)

    # Train.
    trainer.run()

    # Run some code after training.
    if after_training:
        exec(after_training)

train('import tonic.torch\nfrom tasks.swim_in_viscous_spheres import load_env',
      'tonic.torch.agents.PPO(model=ppo_mlp_model(actor_sizes=(256, 256), critic_sizes=(256,256)))',
      'tonic.environments.ControlSuite("swimmer-swim_in_viscous_spheres")',
      name = 'mlp_256_spheres',
      trainer = 'tonic.Trainer(steps=int(5e5),save_steps=int(1e5))')