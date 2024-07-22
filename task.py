import collections
from dm_control.utils import rewards
import dm_control.suite.swimmer as swimmer
from dm_control.rl import control
from dm_control import suite

import numpy as np

from .viscosity_spheres import ViscositySpheres


_SWIM_SPEED = 0.1

class Swim(swimmer.Swimmer):
    """Task to swim forwards at the desired speed."""

    def __init__(self,
        desired_speed=_SWIM_SPEED,
        with_target = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._desired_speed = desired_speed
        self._target_distance = 2.0

        self.viscosity_spheres = ViscositySpheres.random_init(
            spatial_density = 6.,
            size_mean = 0.15,
            size_sigma = 0.3,
            coef_mean = 0.0,
            coef_sigma = 1.0,
        )
        self.with_target = with_target

    def before_step(self, action, physics):
        super().before_step(action, physics)
        self.viscosity_spheres.apply_forces(physics)


    def initialize_episode(self, physics: swimmer.Physics):
        super().initialize_episode(physics)

        if self.with_target:
            # Set the target's position
            target_pos = np.random.uniform(-10, 10, 3)
            target_pos[-1] = 0.05 # z-coord
            physics.named.model.geom_pos["target"][:3] = target_pos
        else:
            # Hide target by setting alpha to 0.
            physics.named.model.mat_rgba["target", "a"] = 1
            physics.named.model.mat_rgba["target_default", "a"] = 1
            physics.named.model.mat_rgba["target_highlight", "a"] = 1

        

    def get_observation(self, physics):
        """Returns an observation of joint angles and body velocities."""
        obs = collections.OrderedDict()
        obs["joints"] = physics.joints()
        obs["body_velocities"] = physics.body_velocities()
        obs["d"] = physics.nose_to_target_dist()
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
    


@swimmer.SUITE.add()
def swim(
    n_links=6,
    desired_speed=_SWIM_SPEED,
    time_limit=swimmer._DEFAULT_TIME_LIMIT,
    random=None,
    environment_kwargs={},
):
    """Returns the Swim task for a n-link swimmer."""
    task = Swim(desired_speed=desired_speed, random=random)

    model_string, assets = swimmer.get_model_and_assets(n_links)
    model_string = task.viscosity_spheres.update_xml(model_string)
    physics = swimmer.Physics.from_xml_string(model_string, assets=assets)
    
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=swimmer._CONTROL_TIMESTEP,
        **environment_kwargs,
    )


def load_env():
    return suite.load("swimmer", "swim", task_kwargs={"random": 1})