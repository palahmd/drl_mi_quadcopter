/*
 * @Author: Wei Luo
 * @Date: 2021-05-21 11:00:08
 * @LastEditors: Wei Luo
 * @LastEditTime: 2021-05-21 16:10:21
 * @Note: Note
 */

#include "mpc_controller.hpp"
#include "quadcopterInit.hpp"

int main(int argc, char *argv[])
{
    auto binaryPath = raisim::Path::setFromArgv(argv[0]);
    raisim::World::setActivationKey(binaryPath.getDirectory() + "\\rsc\\activation.raisim");
    raisim::World world;
    raisim::RaisimServer server(&world);
    world.setTimeStep(timeStep);

    // create raisim objects
    ground = world.addGround();
    robot = world.addArticulatedSystem(
        binaryPath.getDirectory() + "\\rsc\\ITM-quadcopter\\urdf\\ITM-quadcopter.urdf");
    robot->setName("Quadcopter");
    robot->setIntegrationScheme(raisim::ArticulatedSystem::IntegrationScheme::RUNGE_KUTTA_4);

    // define dimension of general coordinates, general velocities, number of rotors and observation vector
    gcDim = robot->getGeneralizedCoordinateDim();
    gvDim = robot->getDOF();
    nRotors = gvDim - 6;
    obDim = 18;

    // initialize containers
    gc.setZero(gcDim);
    gc_init.setZero(gcDim);
    gv.setZero(gvDim);
    gv_init.setZero(gvDim);
    genForces.setZero(gvDim);
    ob.setZero(obDim);
    ob_q.setZero(obDim - 5);

    // initialize state and nominal configuration: [0]-[2]: center of mass, [3]-[6]: quaternions, [7]-[10]: rotors
    gc_init << 0, 0, 0.135, 1, 0, 0, 0, 0.0, 0.0, 0.0, 0.0;
    gv_init << 0, 0, 0, 0, 0, 0, -400 * rpm, 400 * rpm, -400 * rpm, 400 * rpm; // rotor movement for visualization
    robot->setState(gc_init, gv_init);

    // initialize rotor thrusts and conversion matrix for generated forces and torques
    thrusts.setZero(nRotors);
    thrusts2TorquesAndForces << 1, 1, 1, 1,
        rotorPos, -rotorPos, -rotorPos, rotorPos,
        -rotorPos, -rotorPos, rotorPos, rotorPos,
        momConst, -momConst, momConst, -momConst;

    // MPC controller
    MPCAcadosController mpc_controller();
}