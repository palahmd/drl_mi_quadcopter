import os
import numpy as np
import raisimpy as raisim
import time

raisim.World.set_license_file(os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/activation.raisim")
quadcopter_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/quadcopter/Hummingbird/obj/hummingbird.obj"

world = raisim.World()
world.set_time_step(0.001)
server = raisim.RaisimServer(world)
ground = world.add_ground()

dummy_mass_matrix = np.identity(3)

quadcopter = world.add_mesh(quadcopter_urdf_file, 0.72, dummy_mass_matrix, np.array([0, 0, 0.12]), 1)
quadcopter.set_name("Hummingbird")
quadcopter.set_position(0 , 0 , 5)


server.launch_server(8080)


time.sleep(2)
world.integrate1()

while 1:
    world.integrate()

server.kill_server()