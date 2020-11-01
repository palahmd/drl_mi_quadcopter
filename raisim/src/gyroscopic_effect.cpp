#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"
#include "Eigen/Dense"
#include "iostream"


int main(int argc, char* argv[]) {
    auto binaryPath = raisim::Path::setFromArgv(argv[0]);
    raisim::World::setActivationKey( binaryPath.getDirectory() + "\\rsc\\activation.raisim");
    raisim::World world;
    world.setTimeStep(0.002); // what does timestep do and why does the sim. go slower when timestep is changed

    /// create raisim objects
    auto ground = world.addGround();
    auto quadcopter = world.addArticulatedSystem(binaryPath.getDirectory() + "\\rsc\\quadcopter\\ITM_Quadrocopter\\urdf\\ITM_Quadrocopter.urdf");
    quadcopter->setName("Quaddy");

    //quadcopter model parameters
    const double rotorPos = 0.17104913036744201, momentConst = 0.016;
    const double k_f = 6.11*pow(10,-8), k_m = 1.5*pow(10,-9); //in rpm²
    const double rps = 2*M_PI, rpm = rps/60;

    // initialize general coordinates, velocities and number of rotors
    int gcDim = quadcopter->getGeneralizedCoordinateDim();
    int gvDim = quadcopter->getDOF();
    int nRotors = gvDim - 6;

    /// initialize containers
    Eigen::VectorXd gc_init, gv_init, gc, gv, pTarget, pTarget12, vTarget;
    gc.setZero(gcDim); gc_init.setZero(gcDim);
    gv.setZero(gvDim); gv_init.setZero(gvDim);
    pTarget.setZero(gcDim); vTarget.setZero(gvDim); pTarget12.setZero(nRotors);

    /// nominal configuration of quadcopter: [0]-[2]: center of mass, [3]-[6]: quanternions, [7]-[10]: rotors
    gc_init << 0, 0, 10.1433, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    gv_init << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    quadcopter->setState(gc_init, gv_init);

    /// rotor thrusts and equivalent generated forces
    Eigen::VectorXd thrusts;
    Eigen::Matrix4d thrusts2EqForce;
    thrusts.setZero(nRotors);
    thrusts2EqForce << rotorPos, rotorPos, -rotorPos, -rotorPos,
            rotorPos, -rotorPos, -rotorPos, rotorPos,
            momentConst, momentConst, momentConst, momentConst,
            1, 1, 1, 1;
    vTarget = thrusts.cast<double>();
    Eigen::Vector4d testvector;
    testvector << 1, 2, 3, 4;

    std::cout << testvector.segment(0,3) << std::endl;

    /// set initial state and dynamics properties
    quadcopter->setIntegrationScheme(raisim::ArticulatedSystem::IntegrationScheme::SEMI_IMPLICIT);
    quadcopter->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    quadcopter->setGeneralizedForce({0, 0, 0, 0, 0, 0, 0, 0, 0, 0});


    /// launch raisim server for visualization. Can be visualized on raisimUnity
    raisim::RaisimServer server(&world);
    server.launchServer();
    server.focusOn(quadcopter);

    size_t f = 0;
    for (int i=0; i<20000; i++){
//    quadcopter->setExternalForce(1, raisim::ArticulatedSystem::Frame::BODY_FRAME,
//                                 {0, 0, 5}, raisim::ArticulatedSystem::Frame::BODY_FRAME, {0, 0, 0});
//    quadcopter->setExternalForce(2, raisim::ArticulatedSystem::Frame::BODY_FRAME,
//                                   {0, 0, 5}, raisim::ArticulatedSystem::Frame::BODY_FRAME, {0, 0, 0});
//    quadcopter->setExternalForce(3, raisim::ArticulatedSystem::Frame::BODY_FRAME,
//                                   {0, 0, 6}, raisim::ArticulatedSystem::Frame::BODY_FRAME, {0, 0, 0});
//    quadcopter->setExternalForce(4, raisim::ArticulatedSystem::Frame::BODY_FRAME,
//                                   {0, 0, 5}, raisim::ArticulatedSystem::Frame::BODY_FRAME, {0, 0, 0});

        // gyroscopic effect is less than 0.01 Nm
        quadcopter->getState(gc, gv);
        gv[0] = 0, gv[1] = 0, gv[2] = 0, gv[3] = 0, gv[5] = 0;

        if (f>1) {

            gv[6] = 10000*rpm;
            gv[7] = -10000*rpm;
            gv[8] = 10000*rpm;
            gv[9] = -10000*rpm;
            quadcopter->setExternalTorque(0, {0, 0.01, 0});
        }
        quadcopter->setGeneralizedVelocity(gv);
        raisim::MSLEEP(2);
        server.integrateWorldThreadSafe();
        f++;
    }

    server.killServer();
}
