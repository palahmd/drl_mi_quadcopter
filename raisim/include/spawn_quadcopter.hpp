#include "Eigen/Dense"
#include "raisim/World.hpp"

    raisim::ArticulatedSystem* robot;

    // general coordinates, velocities and dynamics
    Eigen::VectorXd gc_init, gv_init, gc, gv, pTarget, pGain, dTarget, dGain vTarget;
    Eigen::Vector4d thrusts, controlThrusts;
    Eigen::Matrix4d thrusts2EqForce;
    int gvDim, gcDim, nRotors;

    raisim::Mat<3,3> rot;
    raisim::Vec<3> pos, linVel, angVel, desiredPos;
    double errorPos;
    
    Eigen::Vector4d EqForce;
    Eigen::Vector3d torque_BaseFrame, force_BaseFrame;
    raisim::Vec<3> torque_WorldFrame, force_WorldFrame;

    
    // quadcopter model parameters
    const double rotorPos = 0.17104913036744201, momentConst = 0.016;
    const double rps = 2 * M_PI, rpm = rps/60;
    const double max_thrust_i = 8;
    const Eigen::Vector4d lavitateThrusts = {1.727/4*9.81, 1.727/4*9.81, 1.727/4*9.81, 1.727/4*9.81};


void updateState(){
    robot->getBasePosition(pos);
    robot->getBaseOrientation(rot);
    robot->getVelocity(0, linVel);
    robot->getAngularVelocity(0, angVel);
}

void calculateThrusts(){
    Eigen::Vector3d hilf1 = desiredPos.e() - pos.e();
    Eigen::Vector3d hilf2 = {1, 1, 1};
    errorPos = hilf1.transpose()*hilf2;

    controlThrusts = 0.01*errorPos*lavitateThrusts;

    thrusts = lavitateThrusts + controlThrusts;
}

void applyThrusts(){
    EqForce = thrusts2EqForce*thrusts;
    torque_BaseFrame = EqForce.segment(0,3);
    force_BaseFrame << 0.0, 0.0, EqForce(3);

    torque_WorldFrame.e() = rot.e() * torque_BaseFrame;
    force_WorldFrame.e() = rot.e() * force_BaseFrame;

    robot->setExternalForce(0, force_WorldFrame);
    robot->setExternalTorque(0, torque_WorldFrame);
}