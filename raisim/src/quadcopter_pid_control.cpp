//
// Created by pala on 25.10.20.
//
#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"
#include "Eigen/Dense"
#include "iostream"

#include "quadcopterInit.hpp"
#include "pid_controller.cpp"
#include "benchmarkCommon.hpp"


int main(int argc, char *argv[]) {
    auto binaryPath = raisim::Path::setFromArgv(argv[0]);
    raisim::World::setActivationKey(binaryPath.getDirectory() + "\\rsc\\activation.raisim");
    raisim::World world;
    raisim::RaisimServer server(&world);
    world.setTimeStep(timeStep);

    /// create raisim objects
    ground = world.addGround(0, "rubber");
    robot = world.addArticulatedSystem(
            binaryPath.getDirectory() + "\\rsc\\ITM-quadcopter\\urdf\\ITM-quadcopter.urdf");
    robot->setName("Quaddy");
    robot->setIntegrationScheme(raisim::ArticulatedSystem::IntegrationScheme::RUNGE_KUTTA_4);

    /// define dimension of general coordinates, general velocities, number of rotors and observation vector
    gcDim = robot->getGeneralizedCoordinateDim();
    gvDim = robot->getDOF();
    nRotors = gvDim - 6;
    obDim = 18;

    /// initialize containers
    gc.setZero(gcDim); gc_init.setZero(gcDim); gv.setZero(gvDim); gv_init.setZero(gvDim);
    genForces.setZero(gcDim); ob.setZero(obDim); ob_q.setZero(obDim - 5);

    /// initialize state and nominal configuration: [0]-[2]: center of mass, [3]-[6]: quaternions, [7]-[10]: rotors
    gc_init << 0, 0, 0.135, 1, 0, 0, 0, 0.0, 0.0, 0.0, 0.0;
    gv_init << 0, 0, 0, 0, 0, 0, -4000 * rpm, 4000 * rpm, -4000 * rpm, 4000 * rpm; // rotor movement for visualization
    robot->setState(gc_init, gv_init);

    /// initialize rotor thrusts and conversion matrix for generated forces and torques
    thrusts.setZero(nRotors);
    thrusts2TorquesAndForces << 1, 1, 1, 1,
            rotorPos, -rotorPos, -rotorPos, rotorPos,
            -rotorPos, -rotorPos, rotorPos, rotorPos,
            momConst, -momConst, momConst, -momConst;

    /// set PID Controller with desired Position for waypoint tracking
    pidController pid(2, 20, 6);
    pid.setTargetPoint(100, 10, 10);

    /// visualize target position
    auto visPoint = server.addVisualSphere("visPoint", 0.25, 0, 0.8, 0);
    visPoint->setPosition(pid.targetPoint.head(3));

    /// launch raisim server for visualization. Can be visualized on raisimUnity
    server.launchServer();
    server.focusOn(robot);
    raisim::MSLEEP(1000); // freeze for 1 sec in the beginning

    /// Integration loop
    auto begin = std::chrono::steady_clock::now();
    loopCount = 5;

    for (i = 0; i < 20000; i++) {
        updateState();
        pid.smallAnglesControl();
        applyThrusts();

        // Loop count for PID controller -> Position Controller controls at every 5th time step
        if (loopCount > 4){
            loopCount = 0;
        }
        loopCount++;
        raisim::MSLEEP(10);
        server.integrateWorldThreadSafe();
    }

    /// Benchmark of the algorithms in the integration loop
    auto end = std::chrono::steady_clock::now();
    raisim::print_timediff(i,begin,end);

    server.killServer();
}

