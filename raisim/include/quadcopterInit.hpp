//
// Created by pala on 02.11.20.
//
// This header initializes all dynamics and model properties and provides basic functions
#ifndef QUADCOPTER_INIT_HPP
#define QUADCOPTER_INIT_HPP
#include "Eigen/Dense"
#include "raisim/World.hpp"

    /// declare raisim world, server and objects
    raisim::Ground* ground;
    raisim::ArticulatedSystem* robot;
    double timeStep = 0.01;
    int i, loopCount;

    /// generalized coordinates, generalized velocities and state
    Eigen::VectorXd gc_init, gv_init, gc, gv;
    int gvDim, gcDim, nRotors, obDim;
    raisim::Mat<3,3> rot;
    raisim::Vec<3> pos, linVel_W, angVel_W, linAcc_W, angAcc_W;
    Eigen::VectorXd ob, ob_q;

    /// thrusts and forces
    Eigen::Vector4d thrusts;
    Eigen::Matrix4d thrusts2TorquesAndForces;
    Eigen::Vector4d torquesAndForces;
    Eigen::Vector3d torques_baseFrame, forces_baseFrame;
    raisim::Vec<3> torques_worldFrame, forces_worldFrame, torques_worldFrame_2, forces_worldFrame_2;
    Eigen::VectorXd genForces;

    /// quadcopter model parameters
    const double rotorPos = 0.17104913036744201, momConst = 0.016;
    const double rps = 2 * M_PI, rpm = rps/60;
    const double g = 9.81, m = 1.727;
    const double hoverThrust = m * g / 4;
    const Eigen::Vector3d inertiaDiagVec = {0.006687, 0.0101, 0.00996};


/** Basic Functions **/
void updateState() {
    robot->getBasePosition(pos);
    robot->getBaseOrientation(rot);
    robot->getVelocity(0, linVel_W);  // bodyIdx: 0 = base
    robot->getAngularVelocity(0, angVel_W);

    /// World Frame accelerations. gc and gv are from the last time step
    linAcc_W = (linVel_W.e() - gv.head(3)) / timeStep;
    angAcc_W = (angVel_W.e() - gv.segment(3, 3)) / timeStep;

    /// get gc and gv from this time step for the next iteration
    robot->getState(gc, gv);

    /// observation vector (later for RL-Algorithm)
    // World Frame position: ob[0]-ob[2], ob_q[0]-ob_q[2]
    for (size_t i = 0; i < 3; i++) {
        ob[i] = pos[i];
        ob_q[i] = pos[i];
    }

    // body velocities: ob[3]-ob[5], ob_q[3]-ob_q[5]
    for (size_t i = 0; i < 3; i++) {
        ob[i + 3] = linVel_W[i];
        ob_q[i + 3] = linVel_W[i];
    }
    // body angular velocities: ob[6]-ob[8], ob_q[6]-ob_q[8]
    for (size_t i = 0; i < 3; i++) {
        ob[i + 6] = angVel_W[i];
        ob_q[i + 6] = angVel_W[i];
    }

    // World Frame rotation Matrix and quaternion: ob[9]-ob[17], ob_q[9]-ob_q[12]
    for (size_t i = 0; i < 9; i++) {
        ob[i + 9] = rot[i];
    }
    for(size_t i=0; i<4; i++) {
        ob_q[i + 9] = gc[i+3];
    }

    /// additional: to maintain rotor velocity in every time step
    // gv.tail(4) << -4000*rpm, 4000*rpm, -4000*rpm, 4000*rpm;
    // robot->setState(gc, gv);
}

void applyThrusts(){
    /// calculate Forces and Torques
    torquesAndForces = thrusts2TorquesAndForces * thrusts;
    forces_baseFrame << 0.0, 0.0, torquesAndForces(0);
    torques_baseFrame = torquesAndForces.tail(3);

    torques_worldFrame.e() = rot.e() * torques_baseFrame;
    forces_worldFrame.e() = rot.e() * forces_baseFrame;

<<<<<<< HEAD
    genForces << forces_worldFrame.e(), torques_worldFrame.e(), 0, 0, 0, 0;
=======
    genForces.head(6) << forces_worldFrame.e(), torques_worldFrame.e();
>>>>>>> master
    robot->setGeneralizedForce(genForces);

    /// this will visualize the applied forces and torques
    // robot->setExternalForce(0, forces_worldFrame);
    // robot->setExternalTorque(0, torques_worldFrame);
}

void applyDragForces(){
<<<<<<< HEAD
    /// TODO
=======
>>>>>>> master
}

#endif //QUADCOPTER_INIT_HPP