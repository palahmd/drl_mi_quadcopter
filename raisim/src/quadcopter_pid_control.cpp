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
    raisim::RaisimServer server(&world);
    world.setTimeStep(timeStep);

    /// create raisim objects
    ground = world.addGround();
    robot = world.addArticulatedSystem(
            binaryPath.getDirectory() + "\\rsc\\quadcopter\\ITM_Quadcopter\\urdf\\ITM_Quadcopter.urdf");
    robot->setName("Quaddy");
    robot->setIntegrationScheme(raisim::ArticulatedSystem::IntegrationScheme::RUNGE_KUTTA_4);

    /// initialize general coordinates, general velocities, number of rotors and observation vector dimension
    gcDim = robot->getGeneralizedCoordinateDim();
    gvDim = robot->getDOF();
    nRotors = gvDim - 6;
    obDim = 18;

    /// initialize containers
    gc.setZero(gcDim);
    gc_init.setZero(gcDim);
    gv.setZero(gvDim);
    gv_init.setZero(gvDim);
    genForces.setZero(gcDim);
    ob.setZero(obDim);
    ob_q.setZero(obDim - 5);
    gc_last.setZero(gcDim);

    /// initialize state and nominal configuration: [0]-[2]: center of mass, [3]-[6]: quaternions, [7]-[10]: rotors
    gc_init << 0, 0, 0.135, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    gv_init << 0, 0, 0, 0, 0, 0, -4000 * rpm, 4000 * rpm, -4000 * rpm, 4000 * rpm; // rotor movement for visualization
    robot->setState(gc_init, gv_init);

    /// rotor thrusts and generated forces and torques
    thrusts.setZero(nRotors);
    thrusts2TorquesAndForces << 1, 1, 1, 1,
            rotorPos, -rotorPos, -rotorPos, rotorPos,
            -rotorPos, -rotorPos, rotorPos, rotorPos,
            momConst, -momConst, momConst, -momConst;

    /// set PID Controller and desired Position for waypoint tracking
    pidController pid(0.5, 50, 1.2);
    pid.setTargetPoint(10, 10, 10);

    /// launch raisim server for visualization. Can be visualized on raisimUnity
    server.launchServer();
    server.focusOn(robot);

    auto visPoint = server.addVisualSphere("visPoint", 0.25, 1, 0, 0);
    visPoint->setPosition(pid.targetPoint.head(3));

    /// Integration loop
    size_t i;
    for (i = 0; i < 200000; i++) {
        updateState();
        pid.smallAnglesController();
        applyThrusts();

        raisim::MSLEEP(2);
        server.integrateWorldThreadSafe();
    }

    server.killServer();
}

