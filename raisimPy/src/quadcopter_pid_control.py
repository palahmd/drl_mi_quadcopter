import os
import numpy as np
import raisimpy as raisim
import time

raisim.World.set_license_file(os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/activation.raisim")
quadcopter_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/quadcopter/ITM_Quadrocopter/urdf/ITM_Quadrocopter.urdf"

world = raisim.World()
world.set_time_step(0.001)
server = raisim.RaisimServer(world)
ground = world.add_ground()

quadcopter = world.add_articulated_system(quadcopter_urdf_file)
quadcopter.set_name("ITM_Quadcopter")
quadcopter_nominal_joint_config = np.array([0.0, 0.0, 0.1433, 1.0, 0.0, 0.0, 0.0, 0, 0, 0, 0])
quadcopter.set_generalized_coordinates(quadcopter_nominal_joint_config)
quadcopter.set_generalized_forces([0.0, 0.0, 0., 0.0, 0.0, 0.0, 0.0, 0.01, 0.01, 0.01, 0.01])
quadcopter.set_control_mode(raisim.ControlMode.FORCE_AND_TORQUE)

server.launch_server(8080)


time.sleep(2)
world.integrate1()
world.integrate2()

for i in range(5000000):
    world.integrate()

server.kill_server()