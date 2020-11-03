//
// Created by pala on 25.10.20.
//

#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"
#include "Eigen/Dense"

#include "quadcopter_init.hpp"
#include "pid_controller.hpp"
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
    robot->setIntegrationScheme(raisim::ArticulatedSystem::IntegrationScheme::RUNGE_KUTTA_4);
    robot->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

    // initialize general coordinates, general velocities, number of rotors and observation vector
    gcDim = robot->getGeneralizedCoordinateDim();
    gvDim = robot->getDOF();
    nRotors = gvDim - 6;
    int obDim = 18;
    ob.setZero(obDim); ob_q.setZero(obDim-5);

    /// initialize containers
    gc.setZero(gcDim); gc_init.setZero(gcDim); gv.setZero(gvDim); gv_init.setZero(gvDim);
    genForces.setZero(gcDim);


    /// initialize state and nominal configuration: [0]-[2]: center of mass, [3]-[6]: quaternions, [7]-[10]: rotors
    gc_init << 0, 0, 0.135, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    gv_init << 0, 0, 0, 0, 0, 0, -4000*rpm, 4000*rpm, -4000*rpm, 4000*rpm; // rotor movement for visualization
    robot->setState(gc_init, gv_init);

    /// rotor thrusts and equivalent generated forces
    thrusts.setZero(nRotors);
    thrusts2TorquesAndForces << rotorPos, -rotorPos, -rotorPos, rotorPos,
            -rotorPos, -rotorPos, rotorPos, rotorPos,
            momentConst, -momentConst, momentConst, -momentConst,
            1, 1, 1, 1;

    /// launch raisim server for visualization. Can be visualized on raisimUnity
    raisim::RaisimServer server(&world);
    server.launchServer();
    server.focusOn(robot);

    /// desired Position
    Eigen::VectorXd desiredPos;
    desiredPos.setZero(obDim-6);
    desiredPos << 0., 0., 0., // target euler angles
                0.,  2.,  10., // target position Vector
                0.,  0.,  0., // target linear Velocity Vector
                0.,  0.,  0.; // target angular Velocity Vector

            /// Visualize desired position
    auto visPoint = server.addVisualSphere("visPoint", 0.25, 1, 0, 0);
    visPoint->setPosition({0, 2, 10});

    for (int i = 0; i < 200000; i++) {

        updateState();
        pidController::calculateThrusts(desiredPos);
        applyThrusts();

        //std::cout << rot << "/n" << std::endl;

        raisim::MSLEEP(2);
        server.integrateWorldThreadSafe();
    }

    server.killServer();
}

