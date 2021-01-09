//
// Created by pala on 17.11.20.
//
#pragma once

#include <stdlib.h>
#include <cstdint>
#include <set>
#include "../../RaisimGymEnv.hpp"
// #include "include/pidController.hpp"
// #include "include/pid_controller.cpp"

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

public:

    explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg, bool visualizable) :
            RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

        /// add objects
        robot_ = world_->addArticulatedSystem(
                resourceDir_ + "/ITM-quadcopter/urdf/ITM-quadcopter.urdf");
        robot_->setName("Quaddy");
        robot_->setIntegrationScheme(raisim::ArticulatedSystem::IntegrationScheme::RUNGE_KUTTA_4);
        world_->addGround(0);

        /// get robot_ data
        gcDim_ = robot_->getGeneralizedCoordinateDim();
        gvDim_ = robot_->getDOF();
        nRotors_ = gvDim_ - 6;

        /// initialize containers
        gc_.setZero(gcDim_);
        gc_init_.setZero(gcDim_);
        gv_.setZero(gvDim_);
        gv_init_.setZero(gvDim_);

        /// nominal configuration of quadcopter: [0]-[2]: center of mass, [3]-[6]: quanternions, [7]-[10]: rotors
        gc_init_ << 0, 0, 0.135, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        gv_init_ << 0, 0, 0, 0, 0, 0, -4000 * rpm, 4000 * rpm, -4000 * rpm, 4000 * rpm;

        /// MUST BE DONE FOR ALL ENVIRONMENTS
        obDim_ = 18; /// convention described on top
        actionDim_ = nRotors_;
        actionMean_.setZero(actionDim_);
        actionStd_.setZero(actionDim_);
        obDouble_.setZero(obDim_);
        obMean_.setZero(obDim_);
        obStd_.setZero(obDim_);

        /// set pd gains
        // pid = pidController(2,10,6);
        // pid.setTargetPoint(100, 10, 10);

        /// action & observation scaling
        actionMean_.setConstant(2.5);
        actionStd_.setConstant(2.0);

        obMean_ << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, /// rotation matrix
                0.0, 0.0, 0.0, /// target position
                Eigen::VectorXd::Constant(6, 0.0); /// body lin/ang vel 6
        obStd_ << Eigen::VectorXd::Constant(9, 0.7), /// rotation matrix
                Eigen::VectorXd::Constant(3, 0.5), /// target position
                Eigen::VectorXd::Constant(3, 3.0), /// linear velocity
                Eigen::VectorXd::Constant(3, 8.0); /// angular velocities


        /// Reward coefficients
        // READ_YAML(double, forwardVelRewardCoeff_, cfg["forwardVelRewardCoeff"])
        // READ_YAML(double, torqueRewardCoeff_, cfg["torqueRewardCoeff"])

        /// visualize if it is the first environment
        if (visualizable_) {
            server_ = std::make_unique<raisim::RaisimServer>(world_.get());
            server_->launchServer();
            server_->focusOn(robot_);
            // auto visPoint = server_->addVisualSphere("visPoint", 0.25, 0, 0.8, 0);
            // visPoint->setPosition(pid.targetPoint.head(3));
            raisim::MSLEEP(1000);
        }
    }

    void init() final {}

    void reset() final {
        robot_->setState(gc_init_, gv_init_);
        updateObservation();
    }

    float step(const Eigen::Ref<EigenVec> &action) final {
        /// action scaling
       //  pid.smallAnglesControl();
        applyThrusts();
        loopCount = 5;
        if (loopCount > 4) loopCount = 0;
        for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
            if (server_) server_->lockVisualizationServerMutex();
            world_->integrate();
            if (server_) server_->unlockVisualizationServerMutex();
        }
        loopCount++;
        updateObservation();

        return 1.0;
    }
    
    
    void updateObservation() {
        robot_->getBasePosition(pos);
        robot_->getBaseOrientation(rot);
        robot_->getVelocity(0, linVel_W);  // bodyIdx: 0 = base
        robot_->getAngularVelocity(0, angVel_W);

        /// World Frame accelerations. gc_ and gv_ are from the last time step
        linAcc_W = (linVel_W.e() - gv_.head(3)) / timeStep;
        angAcc_W = (angVel_W.e() - gv_.segment(3, 3)) / timeStep;

        /// get gc_ and gv_ from this time step for the next iteration
        robot_->getState(gc_, gv_);

        /// observation vector (later for RL-Algorithm)
        // World Frame position: obDouble_[0]-obDouble_[2], ob_q[0]-ob_q[2]
        for (size_t i = 0; i < 3; i++) {
            obDouble_[i] = pos[i];
            ob_q[i] = pos[i];
        }

        // body velocities: obDouble_[3]-obDouble_[5], ob_q[3]-ob_q[5]
        for (size_t i = 0; i < 3; i++) {
            obDouble_[i + 3] = linVel_W[i];
            ob_q[i + 3] = linVel_W[i];
        }
        // body angular velocities: obDouble_[6]-obDouble_[8], ob_q[6]-ob_q[8]
        for (size_t i = 0; i < 3; i++) {
            obDouble_[i + 6] = angVel_W[i];
            ob_q[i + 6] = angVel_W[i];
        }

        // World Frame rotation Matrix and quaternion: obDouble_[9]-obDouble_[17], ob_q[9]-ob_q[12]
        for (size_t i = 0; i < 9; i++) {
            obDouble_[i + 9] = rot[i];
        }
        for(size_t i=0; i<4; i++) {
            ob_q[i + 9] = gc_[i+3];
        }

        /// additional: to maintain rotor velocity in every time step
        // gv_.tail(4) << -4000*rpm, 4000*rpm, -4000*rpm, 4000*rpm;
        // robot_->setState(gc_, gv_);
    }


  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obDouble_.cast<float>();
  }

    void applyThrusts(){
        /// calculate Forces and Torques
        torquesAndForces = thrusts2TorquesAndForces * thrusts;
        forces_baseFrame << 0.0, 0.0, torquesAndForces(0);
        torques_baseFrame = torquesAndForces.tail(3);

        torques_worldFrame.e() = rot.e() * torques_baseFrame;
        forces_worldFrame.e() = rot.e() * forces_baseFrame;

        genForces.head(6) << forces_worldFrame.e(), torques_worldFrame.e();
        robot_->setGeneralizedForce(genForces);

        /// this will visualize the applied forces and torques
        // robot->setExternalForce(0, forces_worldFrame);
        // robot->setExternalTorque(0, torques_worldFrame);
    }
    bool isTerminalState(float& terminalReward) final {
        terminalReward = float(terminalRewardCoeff_);

        terminalReward = 0.f;
        return false;
    }


    // pidController pid;

    double timeStep = 0.01;
    int i, loopCount;

    raisim::Mat<3,3> rot;
    raisim::Vec<3> pos, linVel_W, angVel_W, linAcc_W, angAcc_W;
    Eigen::VectorXd ob_q;

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




private:
  int gcDim_, gvDim_, nRotors_;
  bool visualizable_ = true;
  raisim::ArticulatedSystem* robot_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
  double terminalRewardCoeff_ = -10.;
  double forwardVelRewardCoeff_ = 0., forwardVelReward_ = 0.;
  double torqueRewardCoeff_ = 0., torqueReward_ = 0.;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_, obMean_, obStd_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::set<size_t> baseIndex_;

    
};
}



