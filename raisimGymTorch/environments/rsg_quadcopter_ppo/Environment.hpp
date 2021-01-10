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
        world_->addGround();

        /// get robot data
        gcDim_ = robot_->getGeneralizedCoordinateDim();
        gvDim_ = robot_->getDOF();
        nRotors_ = gvDim_ - 6;
        obDim_ = 18;
        actionDim_ = nRotors_;

        /// initialize containers
        gc_.setZero(gcDim_);
        gc_init_.setZero(gcDim_);
        gv_.setZero(gvDim_);
        gv_init_.setZero(gvDim_);
        genForces.setZero(gvDim_); /// convention described on top
        actionMean_.setZero(actionDim_);
        actionStd_.setZero(actionDim_);
        obDouble_.setZero(obDim_);
        goalPoint_.setZero(obDim_);
        bodyRot.setZero(9);

        /// nominal configuration of quadcopter: [0]-[2]: center of mass, [3]-[6]: quanternions, [7]-[10]: rotors
        gc_init_ << 0, 0, 0.135, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        gv_init_ << 0, 0, 0, 0, 0, 0, -4000 * rpm, 4000 * rpm, -4000 * rpm, 4000 * rpm;

        /// initialize rotor thrusts and conversion matrix for generated forces and torques
        thrusts.setZero(nRotors_);
        thrusts2TorquesAndForces << 1, 1, 1, 1,
                rotorPos, -rotorPos, -rotorPos, rotorPos,
                -rotorPos, -rotorPos, rotorPos, rotorPos,
                momConst, -momConst, momConst, -momConst;

        /// action & observation scaling
        actionMean_.setConstant(g*hoverThrust);
        actionStd_.setConstant(1);
        goalPoint_ << 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

        /// Reward coefficients
        rewards_.initializeFromConfigurationFile (cfg["reward"]);

        /// visualize if it is the first environment
        if (visualizable_) {
            server_ = std::make_unique<raisim::RaisimServer>(world_.get());
            server_->launchServer();
            server_->focusOn(robot_);

            /// visualize target point
            auto visPoint = server_->addVisualSphere("visPoint", 0.25, 0, 0.8, 0);
            visPoint->setPosition(goalPoint_.head(3));

            raisim::MSLEEP(1000);
        }
    }

    void init() final {}

    void reset() final {
        robot_->setState(gc_init_, gv_init_);
        updateObservation();
    }

    float step(const Eigen::Ref<EigenVec> &action) final {
        /// action scaling TODO: Motor model
        thrusts = action.cast<double>();
        thrusts = thrusts.cwiseProduct(actionStd_);
        thrusts += actionMean_;
        applyThrusts();

        for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
            if (server_) server_->lockVisualizationServerMutex();
            world_->integrate();
            if (server_) server_->unlockVisualizationServerMutex();
        }

        updateObservation();

        rewards_.record("position", std::sqrt(bodyPos.squaredNorm()));
        rewards_.record("thrust", thrusts.squaredNorm());
        rewards_.record("orientation", std::acos(bodyRot[8]));
        rewards_.record("angularVelocity", bodyAngVel.squaredNorm());

        return rewards_.sum();
    }
    
    
    void updateObservation() {
        /// get robot state and transform to body frame
        robot_->getBaseOrientation(worldRot);
        robot_->getState(gc_, gv_);
        bodyPos = gc_.head(3);
        bodyRot = worldRot.e().transpose();
        bodyLinVel = bodyRot * gv_.segment(0,3);
        bodyAngVel = bodyRot * gv_.segment(3,3);

        /// observation vector (later for RL-Algorithm)
        // World Frame position: obDouble_[0]-obDouble_[2], ob_q[0]-ob_q[2]
        for (size_t i = 0; i < 3; i++) {
            obDouble_[i] = bodyPos[i];
        }

        // World Frame rotation Matrix and quaternion: obDouble_[9]-obDouble_[17], ob_q[9]-ob_q[12]
        for (size_t i = 0; i < 9; i++) {
            obDouble_[i + 3] = bodyRot[i];
        }

        // body velocities: obDouble_[3]-obDouble_[5], ob_q[3]-ob_q[5]
        for (size_t i = 0; i < 3; i++) {
            obDouble_[i + 12] = bodyLinVel[i];
        }
        // body angular velocities: obDouble_[6]-obDouble_[8], ob_q[6]-ob_q[8]
        for (size_t i = 0; i < 3; i++) {
            obDouble_[i + 15] = bodyAngVel[i];
        }
    }


  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    obDouble_ -= goalPoint_;
    ob = obDouble_.cast<float>();
  }

    void applyThrusts(){
        /// calculate Forces and Torques
        torquesAndForces = thrusts2TorquesAndForces * thrusts;
        forces_baseFrame << 0.0, 0.0, torquesAndForces(0);
        torques_baseFrame = torquesAndForces.tail(3);

        torques_worldFrame.e() = worldRot.e() * torques_baseFrame;
        forces_worldFrame.e() = worldRot.e() * forces_baseFrame;

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

    raisim::Mat<3,3> worldRot;
    Eigen::Vector3d bodyPos, bodyLinVel, bodyAngVel;
    Eigen::VectorXd bodyRot, ob_q;

    Eigen::VectorXd thrusts;
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


  int gcDim_, gvDim_, nRotors_;
  bool visualizable_ = true;
  raisim::ArticulatedSystem* robot_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_;
  double terminalRewardCoeff_ = -10.;
  Eigen::VectorXd obDouble_, goalPoint_;
  Eigen::Vector4d actionMean_, actionStd_;
  std::set<size_t> baseIndex_;
  raisim::Reward rewards_;
    
};
}



