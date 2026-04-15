
import collections

import dm_control as dm
import dm_control.suite.swimmer as swimmer
from dm_control.rl import control
from dm_control.utils import rewards
from dm_control import suite
from dm_control.suite.wrappers import pixels

from utils import *

_SWIM_SPEED = 0.1

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