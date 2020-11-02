#include "Eigen/Dense"
#include "raisim/World.hpp"

    raisim::ArticulatedSystem* robot;

    // general coordinates, velocities and dynamics
    Eigen::VectorXd gc_init, gv_init, gc, gv, pTarget, pTarget12, vTarget;
    Eigen::VectorXd thrusts;
    Eigen::Matrix4d thrusts2EqForce;
    int gvDim, gcDim, nRotors;

    raisim::Mat<3,3> rot;
    raisim::Vec<3> pos, linVel, angVel;
    
    Eigen::Vector4d EqForce;
    Eigen::Vector3d torque_BaseFrame, force_BaseFrame;
    raisim::Vec<3> torque_WorldFrame, force_WorldFrame;
    
    // quadcopter model parameters
    const double rotorPos = 0.17104913036744201, momentConst = 0.016;
    const double rps = 2 * M_PI, rpm = rps/60;


void updateState(){
    robot->getBasePosition(pos);
    robot->getBaseOrientation(rot);
    robot->getVelocity(1, linVel);
    robot->getAngularVelocity(1, angVel);
}

void calculateThrusts(){
    
}

void applyThrusts(){
    EqForce = thrusts2EqForce*thrusts;
    torque_BaseFrame = EqForce.segment(0,3);
    force_BaseFrame << 0.0, 0.0, EqForce(3);

    torque_WorldFrame.e() = rot.e() * torque_BaseFrame;
    force_WorldFrame.e() = rot.e() * force_BaseFrame;

    robot->setExternalForce(1, force_WorldFrame);
    robot->setExternalTorque(1, torque_WorldFrame);
}