/*
 * @Author: Wei Luo
 * @Date: 2021-05-21 11:00:08
 * @LastEditors: Wei Luo
 * @LastEditTime: 2021-05-21 16:10:21
 * @Note: Note
 */

#include "mpc_controller.hpp"
#include "quadcopterInit.hpp"

Eigen::Vector3d ToEulerAngles(Eigen::VectorXd& q) {
    // quaternions inside observation vector start from ob_q[9]
    // ob_q[9] = w, ob_q[10] = x, ob_q[11] = y, ob_q[12] = z
    Eigen::Vector3d angles;

    // roll (x-axis rotation)
    double sinr_cosp = 2 * (q[9] * q[10] + q[11] * q[12]);
    double cosr_cosp = 1 - 2 * (q[10] * q[10] + q[11] * q[11]);
    angles[0] = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    double sinp = 2 * (q[9] * q[11] - q[12] * q[10]);
    if (std::abs(sinp) >= 1)
        angles[1] = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    else
        angles[1] = std::asin(sinp);


    // yaw (z-axis rotation)
    double siny_cosp = 2 * (q[9] * q[12] + q[10] * q[11]);
    double cosy_cosp = 1 - 2 * (q[11] * q[11] + q[12] * q[12]);
    angles[2] = std::atan2(siny_cosp, cosy_cosp);

    return angles;
}

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
    MPCAcadosController mpc_controller(1.727, 0.01);
    mpc_controller.setTargetPoint(0.0, 0.0, 0.5);

    // visualize target position
    auto visPoint = server.addVisualSphere("visPoint", 0.25, 0, 0.8, 0);
    visPoint->setPosition(mpc_controller.targetPoint.head(3));

    // launch raisim server for visualization. Can be visualized on raisimUnity
    server.launchServer();
    server.focusOn(robot);
    raisim::MSLEEP(1000); // freeze for 1 sec in the beginning

    /// Integration loop
    auto begin = std::chrono::steady_clock::now();
    loopCount = 5;

    for (i = 0; i < 10000; i++) {
        updateState();
        Eigen::Vector3d eulerAngles = ToEulerAngles(ob_q);
        std::cout << eulerAngles << std::endl;
        mpc_controller.currentState << pos.e(), eulerAngles, linVel_W.e();
        // mpc_controller.currentState << pos.e(), ob_q[9], ob_q[10], ob_q[11], ob_q[12], linVel_W.e();
        // mpc_controller.solvingACADOS(thrusts2TorquesAndForces, thrusts);
        // applyThrusts();

        // raisim::MSLEEP(10);
        // server.integrateWorldThreadSafe();
    }


    /// Benchmarking the algorithms within the integration loop
    auto end = std::chrono::steady_clock::now();
    raisim::print_timediff(i, begin, end);

    server.killServer();
}