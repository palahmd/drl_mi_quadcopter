/*
 * @Author: Wei Luo
 * @Date: 2021-05-21 11:00:08
 * @LastEditors: Wei Luo
 * @LastEditTime: 2021-05-21 13:07:12
 * @Note: Note
 */

#include "mpc_controller.hpp"
#include "quadcopterInit.hpp"

int main(int argc, char *argv[])
{
    // auto binaryPath = raisim::Path::setFromArgv(argv[0]);
    // raisim::World::setActivationKey(binaryPath.getDirectory() + "\\rsc\\activation.raisim");
    // raisim::World world;
    // raisim::RaisimServer server(&world);
    // world.setTimeStep(timeStep);

    // /// create raisim objects
    // ground = world.addGround();
    // robot = world.addArticulatedSystem(
    //     binaryPath.getDirectory() + "\\rsc\\ITM-quadcopter\\urdf\\ITM-quadcopter.urdf");
    // robot->setName("Quadcopter");
    // robot->setIntegrationScheme(raisim::ArticulatedSystem::IntegrationScheme::RUNGE_KUTTA_4);

    // /// define dimension of general coordinates, general velocities, number of rotors and observation vector
    // gcDim = robot->getGeneralizedCoordinateDim();
    // gvDim = robot->getDOF();
    // nRotors = gvDim - 6;
    // obDim = 18;

    MPCAcadosController mpc_controller();
}