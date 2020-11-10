//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <cstdint>
#include <set>
#include "../../RaisimGymEnv.hpp"

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

 public:

  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

    /// add objects
    quadcopter_ = world_->addArticulatedSystem(resourceDir_+"/quadcopter/ITM_Quadcopter/urdf/ITM_Quadcopter.urdf");
    quadcopter_->setName("ITM Quadcopter");
    quadcopter_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    world_->addGround(-1);

    /// get robot data
    gcDim_ = quadcopter_->getGeneralizedCoordinateDim();
    gvDim_ = quadcopter_->getDOF();
    nRotors_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nRotors_);

    /// nominal configuration of quadcopter: [0]-[2]: center of mass, [3]-[6]: quanternions, [7]-[10]: rotors
    gc_init_ << 0, 0, 0.1433, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero(); jointPgain.tail(nRotors_).setConstant(4.0);
    jointDgain.setZero(); jointDgain.tail(nRotors_).setConstant(1.0);
    quadcopter_->setPdGains(jointPgain, jointDgain);
    quadcopter_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 18; /// convention described on top
    actionDim_ = nRotors_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);
    obMean_.setZero(obDim_); obStd_.setZero(obDim_);

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
    READ_YAML(double, forwardVelRewardCoeff_, cfg["forwardVelRewardCoeff"])
    READ_YAML(double, torqueRewardCoeff_, cfg["torqueRewardCoeff"])

    /// indices of links that should not make contact with ground
    baseIndex_.insert(quadcopter_->getBodyIdx("base_link"));

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
      server_->focusOn(quadcopter_);
    }
  }

  void init() final { }

  void reset() final {
    quadcopter_->setState(gc_init_, gv_init_);
    updateObservation();
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    /// action scaling
    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(nRotors_) = pTarget12_;

    quadcopter_->setPdTarget(pTarget_, vTarget_);

    for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
      if(server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if(server_) server_->unlockVisualizationServerMutex();
    }

    updateObservation();

    torqueReward_ = torqueRewardCoeff_ * quadcopter_->getGeneralizedForce().squaredNorm();
    forwardVelReward_ = forwardVelRewardCoeff_ * std::min(4.0, bodyLinearVel_[0]);
    return torqueReward_ + forwardVelReward_;
  }

  void updateObservation() {
    quadcopter_->getState(gc_, gv_);
    raisim::Vec<4> quat;
    raisim::Mat<3,3> rot;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);

    obDouble_ << gc_[2], /// body height
        rot.e().row(2).transpose(), /// body orientation
        gc_.tail(12), /// joint angles
        bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity
        gv_.tail(12); /// joint velocity
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obDouble_.cast<float>();
  }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = float(terminalRewardCoeff_);

    /// if the contact body is not feet
    for(auto& contact: quadcopter_->getContacts())
      if(baseIndex_.find(contact.getlocalBodyIndex()) == baseIndex_.end())
        return true;

    terminalReward = 0.f;
    return false;
  }

 private:
  int gcDim_, gvDim_, nRotors_;
  bool visualizable_ = false;
  raisim::ArticulatedSystem* quadcopter_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
  double terminalRewardCoeff_ = -10.;
  double forwardVelRewardCoeff_ = 0., forwardVelReward_ = 0.;
  double torqueRewardCoeff_ = 0., torqueReward_ = 0.;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_, obMean_, obStd_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::set<size_t> baseIndex_;
};
}

