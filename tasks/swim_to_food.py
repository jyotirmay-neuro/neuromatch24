import numpy as np
from dm_control import suite
import dm_control.suite.swimmer as swimmer
from dm_control.rl import control
from lxml import etree


def update_xml_with_food_zone(xml_string: str):
    tree = etree.fromstring(xml_string)
    worldbody = tree.find('./worldbody')

    food_zone = etree.Element("geom", name="food_zone", type="box", pos="0 0 0.05", size="0.05 0.05 0.05", material="target")

    # Find the old element
    old_element = worldbody.find(".//geom[@name='target']")
    if old_element is not None:
        # Remove the old element
        worldbody.remove(old_element)
    # Add the new element
    worldbody.append(food_zone)

    return etree.tostring(tree, pretty_print=True)


       

class Swim(swimmer.Swimmer):
    def __init__(self, arena_size=(1, 1)):
        self.arena_size = arena_size
        self.food_zone_size = min(arena_size) / 10
        self.food_zone_pos = self._random_food_zone()
        super().__init__()

    def _random_food_zone(self):
        x = np.random.uniform(-self.arena_size[0] / 2 + self.food_zone_size / 2,
                              self.arena_size[0] / 2 - self.food_zone_size / 2)
        y = np.random.uniform(-self.arena_size[1] / 2 + self.food_zone_size / 2,
                              self.arena_size[1] / 2 - self.food_zone_size / 2)
        return np.array([x, y])

    def initialize_episode(self, physics):
        super().initialize_episode(physics)
        self.food_zone_pos = self._random_food_zone()
        physics.named.data.geom_xpos['food_zone'][:2] = self.food_zone_pos

    def get_observation(self, physics):
        obs = {}
        obs['joints'] = physics.joints()
        obs['position'] = physics.named.data.geom_xpos['nose'][:2]
        obs['food_zone'] = self.food_zone_pos
        obs['smell_strength'] = self._smell_strength(obs['position'])
        obs['to_target'] = physics.nose_to_target()
        obs['body_velocities'] = physics.body_velocities()
        return obs

    def _smell_strength(self, position):
        distance = np.linalg.norm(position - self.food_zone_pos)
        return np.exp(-distance)

    def get_reward(self, physics):
        position = physics.named.data.geom_xpos['nose'][:2]
        smell_strength = self._smell_strength(position)
        vel = physics.body_velocities()
        reward = smell_strength + 0.1 * vel
        if np.linalg.norm(position - self.food_zone_pos) > self.food_zone_size:
            reward -= 0.1 * np.exp(np.linalg.norm(position - self.food_zone_pos))
        return reward

@swimmer.SUITE.add()
def swim_to_food(
    n_links=6,
    time_limit=swimmer._DEFAULT_TIME_LIMIT,
    random=None ):
    """Returns the Swim task for a n-link swimmer."""
    task = Swim(arena_size=(10, 10))

    model_string, assets = swimmer.get_model_and_assets(n_links)
    model_string = update_xml_with_food_zone(model_string)
    physics = swimmer.Physics.from_xml_string(model_string, assets=assets)
    
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        control_timestep=swimmer._CONTROL_TIMESTEP,
        **environment_kwargs,
    )


def load_env(**kwargs):
    return suite.load("swimmer", "swim_to_food")
