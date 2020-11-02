#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"
#include "Eigen/Dense"
#include "spawn_quadcopter.hpp"
#include "iostream"


int main(int argc, char *argv[]) {
    auto binaryPath = raisim::Path::setFromArgv(argv[0]);
    raisim::World::setActivationKey(binaryPath.getDirectory() + "\\rsc\\activation.raisim");
    raisim::World world;
    world.setTimeStep(0.002); // what does timestep do and why does the sim. go slower when timestep is changed

    /// create raisim objects
    auto ground = world.addGround();
    robot = world.addArticulatedSystem(
            binaryPath.getDirectory() + "\\rsc\\quadcopter\\ITM_Quadrocopter\\urdf\\ITM_Quadrocopter.urdf");
    robot->setName("Quaddy");
    robot->setIntegrationScheme(raisim::ArticulatedSystem::IntegrationScheme::SEMI_IMPLICIT);
    robot->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

    // initialize general coordinates, velocities and number of rotors
    gcDim = robot->getGeneralizedCoordinateDim();
    gvDim = robot->getDOF();
    nRotors = gvDim - 6;

    /// initialize containers
    gc.setZero(gcDim); gc_init.setZero(gcDim); gv.setZero(gvDim); gv_init.setZero(gvDim);
    pTarget.setZero(gcDim); vTarget.setZero(gvDim); pTarget12.setZero(nRotors);

    /// initialize state and nominal configuration: [0]-[2]: center of mass, [3]-[6]: quanternions, [7]-[10]: rotors
    /// also possible to set a reset() function
    gc_init << 0, 0, 0.135, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    gv_init << 0, 0, 0, 0, 0, 0, -2000*rpm, 2000*rpm, -2000*rpm, 2000*rpm; // rotor movement for visualization
    robot->setState(gc_init, gv_init);

    /// rotor thrusts and equivalent generated forces
    thrusts.setZero(nRotors);
    thrusts2EqForce << rotorPos, -rotorPos, -rotorPos, rotorPos,
            -rotorPos, -rotorPos, rotorPos, rotorPos,
            momentConst, -momentConst, momentConst, -momentConst,
            1, 1, 1, 1;

    /// launch raisim server for visualization. Can be visualized on raisimUnity
    raisim::RaisimServer server(&world);
    server.launchServer();
    server.focusOn(robot);


    for (int i = 0; i < 200000; i++) {
        raisim::MSLEEP(2);
        server.integrateWorldThreadSafe();
    }

    server.killServer();
}

