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
#include <iostream>
#include <math.h>

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
        genForces_.setZero(gvDim_); /// convention described on top
        actionMean_.setZero(actionDim_);
        actionStd_.setZero(actionDim_);
        obDouble_.setZero(obDim_);
        goalPoint_.setZero(obDim_);

        /// nominal configuration of quadcopter: [0]-[2]: center of mass, [3]-[6]: quanternions, [7]-[10]: rotors
        gc_init_ << 0.0, 0.0, 0.135, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        // gv_init_ << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4000 * rpm_, 4000 * rpm_, -4000 * rpm_, 4000 * rpm_;

        /// initialize rotor thrusts and conversion matrix for generated forces and torques
        thrusts_.setZero(nRotors_);
        thrusts2TorquesAndForces_ << 1, 1, 1, 1,
                rotorPos_, -rotorPos_, -rotorPos_, rotorPos_,
                -rotorPos_, -rotorPos_, rotorPos_, rotorPos_,
                momConst_, -momConst_, momConst_, -momConst_;

        /// action & observation scaling
        actionMean_.setConstant(g_*hoverThrust_);
        actionStd_.setConstant(1);
        goalPoint_ << 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

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
        thrusts_ = action.cast<double>();
        thrusts_ = thrusts_.cwiseProduct(actionStd_);
        thrusts_ += actionMean_;
        // applyThrusts();

        for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
            if (server_) server_->lockVisualizationServerMutex();
            world_->integrate();
            if (server_) server_->unlockVisualizationServerMutex();
        }

        // updateObservation();

        rewards_.record("position", std::sqrt(bodyPos_.squaredNorm()));
        rewards_.record("thrust", thrusts_.squaredNorm());
        rewards_.record("orientation", std::acos(bodyRot_(2,1)));
        rewards_.record("angularVelocity", bodyAngVel_.squaredNorm());

        return rewards_.sum();
    }
    
    
    void updateObservation() {
        /// get robot state and transform to body frame
        robot_->getBaseOrientation(worldRot_);
        robot_->getState(gc_, gv_);
        bodyPos_ = gc_.head(3);
        bodyRot_ = worldRot_.e().transpose();
        bodyLinVel_ = bodyRot_ * gv_.segment(0,3);
        bodyAngVel_ = bodyRot_ * gv_.segment(3,3);

        /// observation vector (later for RL-Algorithm)
        // World Frame position: obDouble_[0]-obDouble_[2], ob_q[0]-ob_q[2]
        for (size_t i = 0; i < 3; i++) {
            obDouble_[i] = bodyPos_[i];
        }

        // World Frame rotation Matrix and quaternion: obDouble_[9]-obDouble_[17], ob_q[9]-ob_q[12]
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j<3; j++){
                obDouble_[i + j + 3] = bodyRot_(i,j);
            }
        }

        // body velocities: obDouble_[3]-obDouble_[5], ob_q[3]-ob_q[5]
        for (size_t i = 0; i < 3; i++) {
            obDouble_[i + 12] = bodyLinVel_[i];
        }
        // body angular velocities: obDouble_[6]-obDouble_[8], ob_q[6]-ob_q[8]
        for (size_t i = 0; i < 3; i++) {
            obDouble_[i + 15] = bodyAngVel_[i];
        }
    }


  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    obDouble_ -= goalPoint_;
    ob = obDouble_.cast<float>();
  }

    void applyThrusts(){
        /// calculate Forces and Torques
        torquesAndForces_ = thrusts2TorquesAndForces_ * thrusts_;
        forces_baseFrame_ << 0.0, 0.0, torquesAndForces_(0);
        torques_baseFrame_ = torquesAndForces_.tail(3);

        torques_worldFrame_.e() = worldRot_.e() * torques_baseFrame_;
        forces_worldFrame_.e() = worldRot_.e() * forces_baseFrame_;

        genForces_.head(6) << forces_worldFrame_.e(), torques_worldFrame_.e();
        robot_->setGeneralizedForce(genForces_);

        /// this will visualize the applied forces and torques
        // robot->setExternalForce(0, forces_worldFrame_);
        // robot->setExternalTorque(0, torques_worldFrame_);
    }
    bool isTerminalState(float& terminalReward) final {
        terminalReward = float(terminalRewardCoeff_);

        terminalReward = 0.f;
        return false;
    }


private:

  raisim::Mat<3,3> worldRot_;
  Eigen::Vector3d bodyPos_, bodyLinVel_, bodyAngVel_;
  Eigen::Matrix3d bodyRot_;

  Eigen::VectorXd thrusts_;
  Eigen::Matrix4d thrusts2TorquesAndForces_;
  Eigen::Vector4d torquesAndForces_;
  Eigen::Vector3d torques_baseFrame_, forces_baseFrame_;
  raisim::Vec<3> torques_worldFrame_, forces_worldFrame_;
  Eigen::VectorXd genForces_;

  /// quadcopter model parameters
  const double rotorPos_ = 0.17104913036744201, momConst_ = 0.016;
  const double rps_ = 2 * M_PI, rpm_ = rps_/60;
  const double g_ = 9.81, m_ = 1.727;
  const double hoverThrust_ = m_ * g_ / 4;

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



