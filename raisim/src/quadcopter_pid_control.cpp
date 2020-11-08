//
// Created by pala on 25.10.20.
//

#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"
#include "Eigen/Dense"

#include "quadcopterInit.hpp"
#include "pid_controller.cpp"
#include "iostream"


int main(int argc, char *argv[]) {
    auto binaryPath = raisim::Path::setFromArgv(argv[0]);
    raisim::World::setActivationKey(binaryPath.getDirectory() + "\\rsc\\activation.raisim");
    raisim::World world;
    world.setTimeStep(timeStep);

    /// create raisim objects
    ground = world.addGround();
    robot = world.addArticulatedSystem(
            binaryPath.getDirectory() + "\\rsc\\quadcopter\\ITM_Quadcopter\\urdf\\ITM_Quadcopter.urdf");
    robot->setName("Quaddy");
    robot->setIntegrationScheme(raisim::ArticulatedSystem::IntegrationScheme::RUNGE_KUTTA_4);
    robot->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

    /// initialize general coordinates, general velocities, number of rotors and observation vector
    gcDim = robot->getGeneralizedCoordinateDim();
    gvDim = robot->getDOF();
    nRotors = gvDim - 6;
    obDim = 18;

    /// initialize containers
    gc.setZero(gcDim); gc_init.setZero(gcDim); gv.setZero(gvDim); gv_init.setZero(gvDim);
    genForces.setZero(gcDim); ob.setZero(obDim); ob_q.setZero(obDim-5);

    /// initialize state and nominal configuration: [0]-[2]: center of mass, [3]-[6]: quaternions, [7]-[10]: rotors
    gc_init << 0, 0, 0.135, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    gv_init << 0, 0, 0, 0, 0, 0, -4000*rpm, 4000*rpm, -4000*rpm, 4000*rpm; // rotor movement for visualization
    robot->setState(gc_init, gv_init);

    /// rotor thrusts and equivalent generated forces
    thrusts.setZero(nRotors);
    thrusts2TorquesAndForces << 1,           1,          1,         1,
                                rotorPos,   -rotorPos,  -rotorPos,  rotorPos,
                                -rotorPos,  -rotorPos,  rotorPos,   rotorPos,
                                momConst,   -momConst,  momConst,   -momConst;


    /// launch raisim server for visualization. Can be visualized on raisimUnity
    raisim::RaisimServer server(&world);
    server.launchServer();
    server.focusOn(robot);


    /// set desired Position for waypoint tracking
    Eigen::VectorXd desiredPos;
    desiredPos.setZero(obDim-6);
    desiredPos << 0., 0., 10., // target position
                0.,  0.,  0., // target linear velocity
                0.,  0.,  0., // target angular velocity
                0.,  0.,  0.; // target euler Angles

    /// Visualize desired position
    auto visPoint = server.addVisualSphere("visPoint", 0.25, 1, 0, 0);
    visPoint->setPosition(desiredPos.head(3));


    for (int i = 0; i < 200000; i++) {
        updateState();
        pidController::calculateThrusts(desiredPos);
        applyThrusts();

        raisim::MSLEEP(2);
        server.integrateWorldThreadSafe();
    }

    server.killServer();
}

