import os
import numpy as np
import raisimpy as raisim
import time

raisim.World.set_license_file(os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/activation.raisim")
quadcopter_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/quadcopter/Hummingbird/urdf/hummingbird.urdf"
dummy_inertia = np.zeros([3, 3])
np.fill_diagonal(dummy_inertia, 0.1)

world = raisim.World()
world.set_time_step(0.001)
server = raisim.RaisimServer(world)
ground = world.add_ground()

anymal = world.add_articulated_system(anymal_urdf_file)
anymal.set_name("Hummingbird")
anymal_nominal_joint_config = np.array([0, -1.5, 0.54, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8,
                                        -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8])
anymal.set_generalized_coordinates(anymal_nominal_joint_config)
anymal.set_pd_gains(200*np.ones([18]), np.ones([18]))
anymal.set_pd_targets(anymal_nominal_joint_config, np.zeros([18]))


server.launch_server(8080)

for i in range(5):
    for j in range(5):
        object_type = (i + j*6) % 5

        if object_type == 0:
            obj = world.add_mesh(monkey_file, 5.0, dummy_inertia, np.array([0, 0, 0]), 0.3)
        elif object_type == 1:
            obj = world.add_cylinder(0.2, 0.3, 2.0)
        elif object_type == 2:
            obj = world.add_capsule(0.2, 0.3, 2.0)
        elif object_type == 3:
            obj = world.add_box(0.4, 0.4, 0.4, 2.0)
        else:
            obj = world.add_sphere(0.3, 2.0)

        obj.set_position(i-2.5, j-2.5, 5)

time.sleep(2)
world.integrate1()

### get dynamic properties
# mass matrix
mass_matrix = anymal.get_mass_matrix()

# non-linear term (gravity+coriolis)
non_linearities = anymal.get_non_linearities()

# non-linear term (gravity+coriolis)
non_linearities = anymal.get_non_linearities()

for i in range(50000):
    world.integrate()

server.kill_server()
