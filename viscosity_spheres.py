from dataclasses import dataclass
from lxml import etree
import numpy as np

@dataclass
class VscSphereP:
    coeff: float
    pos: tuple[float,float, float]
    size: float = .1

class ViscositySpheres:
    """
    Example of use:
    ```
    viscosity_spheres = ViscositySpheres([
        VscSphereP(0.2, (.5, .5, .05)),
        VscSphereP(-0.6, (.1, .1, .05)),
    ])
    ...


    class Swim(swimmer.Swimmer):
        def before_step(self, action, physics):
            super().before_step(action, physics)
            viscosity_spheres.apply_forces(physics)
    ...

    model_string, assets = swimmer.get_model_and_assets(n_links)
    model_string = viscosity_spheres.update_xml(model_string)
    ```


    """

    @staticmethod
    def random_init(
        spatial_density: float = 6.,
        size_mean: float = 0.15,
        size_sigma: float =0.3,
        coef_mean: float = 0.0,
        coef_sigma: float = 1.0,
    ):
        # spacial_dencity - how many spheres per 1^2 area
        # scene 10x10 (?)

        spheres_n = int(spatial_density*100)

        x = np.random.uniform(-10, 10, spheres_n)
        y = np.random.uniform(-10, 10, spheres_n)
        z = np.ones((spheres_n,))*0.06
        coefs = np.random.normal(coef_mean, coef_sigma, spheres_n)
        sizes = np.abs(np.random.normal(size_mean, size_sigma, spheres_n))
        spheres = [ VscSphereP(c, (x,y,z), s) for c, x, y, z, s in zip(coefs, x, y, z, sizes) ]
        return ViscositySpheres(spheres)

    def __init__(self, spheres: list[VscSphereP]):
        self._spheres = spheres

    def update_xml(self, xml_string: str):
        tree = etree.fromstring(xml_string)
        root = tree.find('.')

        # Create the new <asset> section
        asset_element = etree.Element('asset')
        # Create the new <material> elements
        etree.SubElement(asset_element, 'material', name='viscosity', rgba='1 1 1 0.2', specular='0', shininess='0')
        # Append the <asset> section to the root
        root.append(asset_element)

        worldbody = tree.find('./worldbody')

        max_coeff = max([ abs(s.coeff) for s in self._spheres ] + [1.])
        for i, s in enumerate(self._spheres):
            _a = abs(s.coeff) / (max_coeff*2)
            rgba = f"1 0 0 {_a}" if s.coeff < 0 else f"0 1 0 {_a}"
            name = f'viscosity_ball_{i}'
            pos = f"{s.pos[0]} {s.pos[1]} {s.pos[2]}"
            size = f"{s.size}"
            etree.SubElement(worldbody, 'geom', name=name, type="sphere", pos=pos, size=size, material="viscosity", rgba=rgba)
        return etree.tostring(tree, pretty_print=True)

       

    def apply_forces(self, physics):
        body_names = physics.named.data.xpos.axes[0].names

        for body_name in body_names:
            # interested in swimmer body parts only
            if body_name != "head" and not body_name.startswith("segment"):
                continue
            body_pos = physics.named.data.xpos[body_name]


            for s in self._spheres:
                d_body = np.sqrt(((np.array(s.pos) - body_pos)**2).sum())
                if d_body < s.size:
                    linear_v = physics.named.data.cvel[body_name][
                        :3
                    ]  # [:3] for linar vel/forces only
                    viscous_force = -s.coeff*s.size/(d_body + 0.000001)  * linear_v
                    physics.named.data.xfrc_applied[body_name][:3] += viscous_force
                    # physics.named.data.xfrc_applied[body_name][:3] += 0.001
