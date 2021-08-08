//
// Created by pala on 17.11.20.
//
#pragma once

#include <stdlib.h>
#include <random>
#include <cstdint>
#include <set>
#include "../../RaisimGymEnv.hpp"
#include <cmath>
#include "Eigen/Dense"

namespace raisim {

    /** This environment class implements n equally setup parallel environmets. Random values are generated independently
     *  for each environment.
     **/
    class ENVIRONMENT : public RaisimGymEnv {

    public:

        explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg, bool visualizable) :
        RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable), normDist_(0., 1.), gen_(rd_()) {

            /// create world
            world_ = std::make_unique<raisim::World>();

            /// add objects
            /// quadcopter properties are defined in the urdf file
            robot_ = world_->addArticulatedSystem(
                    resourceDir_ + "/ITM-quadcopter/urdf/ITM-quadcopter.urdf");
            robot_->setName("ITM-Quadcopter");
            robot_->setIntegrationScheme(raisim::ArticulatedSystem::IntegrationScheme::RUNGE_KUTTA_4);
            world_->addGround(-15);

            /// get robot data
            gcDim_ = robot_->getGeneralizedCoordinateDim();
            gvDim_ = robot_->getDOF();
            nRotors_ = gvDim_ - 6;
            /** observation space: 18 entries
             *                      position: 3 entr.
             *                      orientation: 9 entr. (orientation matrix R)
             *                      velocity: 3 entr.
             *                      angular velocity: 3 entr.
             *
             *  add. entries:       orientation expressed in quaternions: 4 entr.
             **/
            obDim_ = 22;
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
            targetPoint_.setZero(obDim_);

            /// nominal configuration of the quadcopter: [0]-[2]: center of mass, [3]-[6]: quanternions, [7]-[10]: rotors
            gc_init_ << 0.0, 0.0, 0.135, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
            /// constant rotor velocity for visualization purposes
            gv_init_ << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -700 * rpm_, 700 * rpm_, -700 * rpm_, 700 * rpm_;

            /// initialize rotor thrusts_ and conversion matrix for generated forces and torques
            thrusts_.setZero(nRotors_);
            controlThrusts_.setZero(nRotors_);
            /// conversion matrix - converts the control input to generated force and torques from the motor
            thrusts2TorquesAndForces_ << 1, 1, 1, 1,
            rotorPos_, -rotorPos_, -rotorPos_, rotorPos_,
            -rotorPos_, -rotorPos_, rotorPos_, rotorPos_,
            momConst_, -momConst_, momConst_, -momConst_;

            /** action scaling:
             *  action is scaled down to [-1,1] for the training of the neural network. Thus, the neural network
             *  determines the action within a range of [-1,1]
             **/
            actionMean_.setConstant(hoverThrust_);
            actionStd_.setConstant(0.5 * hoverThrust_);

            /// indices of links that should not make contact with ground - all links rotors
            bodyIndices_.insert(robot_->getBodyIdx("rotor_0"));
            bodyIndices_.insert(robot_->getBodyIdx("rotor_1"));
            bodyIndices_.insert(robot_->getBodyIdx("rotor_2"));
            bodyIndices_.insert(robot_->getBodyIdx("rotor_3"));

            /// Reward coefficients
            rewards_.initializeFromConfigurationFile(cfg["reward"]);

            /// visualize if it is the first environment
            if (visualizable_) {
                server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                server_->launchServer();
                server_->focusOn(robot_);
                visPoint = server_->addVisualSphere("visPoint", 0.2, 0.8, 0, 0);
            }
        }

        void init() final {}

        void reset() final {
            /// set random target point or state
            setRandomTargets(10, 1);
            //setRandomStates(0, true, 2, 4);
            //setTarget(5.77, 5.77, 5.77);

            robot_->setState(gc_init_, gv_init_);
            updateObservation();

            if (visualizable_) server_->focusOn(robot_);
        }

        float step(const Eigen::Ref<EigenVec> &action) final {
            /// the control signal is delayed with one control time step to model the delay on the real quadcopter
            for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
                if (server_) server_->lockVisualizationServerMutex();
                /// delayes the rotor thrusts with a simple linear motor model over all simulation time steps
                for (int i = 0; i < 4; i++) {
                    double omega_des = std::sqrt(controlThrusts_[i] / thrConst);
                    double omega = std::sqrt(thrusts_[i] / thrConst);
                    double delta_omega = motConst * (omega_des - omega);
                    omega += delta_omega * simulation_dt_;
                    thrusts_[i] = thrConst * std::pow(omega, 2);
                }
                /// apply rotor thrusts
                applyThrusts();
                world_->integrate();
                if (server_) server_->unlockVisualizationServerMutex();
            }

            /// get the control signal from the PID-controller in python
            controlThrusts_ = action.cast<double>();

            /// scale action down to [-1, 1] to limit the thrust of the quadcopter as is the case in the real quadcopter.
            /// Should work better than clipping, does at least for the pid controller
            double max_scale = controlThrusts_.maxCoeff();
            double min_scale = controlThrusts_.minCoeff();

            if ((max_scale > 1) || (min_scale < -1)) {
                if (std::abs(min_scale) < std::abs(max_scale)) {
                    controlThrusts_ /= max_scale;
                } else {
                    controlThrusts_ /= min_scale;
                }
            }

            /// normedControlThrust_ is used for the reward function
            normedControlThrusts_ = controlThrusts_;

            /// scale up bounded action input to actual rotor control signals
            controlThrusts_ = controlThrusts_.cwiseProduct(actionStd_);
            controlThrusts_ += actionMean_;

            updateObservation();

            /// relAbsPosition is used for the reward function
            relativeAbsPosition = (targetPoint_.head(3) - bodyPos_).norm();

            /* set a zone (sphere near the target) with higher rewards.
            if (relativeAbsPosition < 1){
                relPositionReward = 1 / (0.5+relativeAbsPosition);
            }
            else{
                relPositionReward = 0;
            }*/

            /// reward function
            rewards_.record("position", relativeAbsPosition);
            rewards_.record("thrust", normedControlThrusts_.norm());
            rewards_.record("orientation", std::acos(worldRot_[8]));
            rewards_.record("angularVelocity", bodyAngVel_.norm());

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
            robot_->getBaseOrientation(quat_);

            /// observation vector (later for RL-Algorithm)
            for (size_t i = 0; i < 3; i++) {
                obDouble_[i] = bodyPos_[i];
            }

            for (size_t i = 0; i < 3; i++) {
                for (size_t j = 0; j<3; j++){
                    obDouble_[j + (i+1)*3] = bodyRot_(i,j);
                }
            }

            for (size_t i = 0; i < 3; i++) {
                obDouble_[i + 12] = bodyLinVel_[i];
            }

            for (size_t i = 0; i < 3; i++) {
                obDouble_[i + 15] = bodyAngVel_[i];
            }

            /// quaternion is needed for the PID-controller in Python
            for (size_t i = 0; i < 4; i++) {
                obDouble_[i + 18] = quat_[i];
            }
        }

        void observe(Eigen::Ref<EigenVec> ob) final {
            /// convert it to float
            obDouble_ -= targetPoint_;
            ob = obDouble_.cast<float>();
        }

        bool isTerminalState(float &terminalReward) final {
            terminalReward = float(terminalRewardCoeff_);

            for (auto &contact: robot_->getContacts()) {
                if (bodyIndices_.find(contact.getlocalBodyIndex()) == bodyIndices_.end()) {
                    loopCount_++;
                    updateTarget = false;
                    return true;
                }
            }

            for (int i = 0; i < 3; i++) {
                if (bodyPos_[i] > 20) {
                    loopCount_++;
                    updateTarget = false;
                    return true;
                }
            }
            terminalReward = 0.f;
            return false;
        }

        void applyThrusts() {
            /// calculate forces and torques w.r.t. the center of mass
            torquesAndForces_ = thrusts2TorquesAndForces_ * thrusts_;
            forces_baseFrame_ << 0.0, 0.0, torquesAndForces_(0);
            torques_baseFrame_ = torquesAndForces_.tail(3);

            torques_worldFrame_.e() = worldRot_.e() * torques_baseFrame_;
            forces_worldFrame_.e() = worldRot_.e() * forces_baseFrame_;

            /// apply forces and torques to the quadcopter
            genForces_.head(3) = forces_worldFrame_.e();
            genForces_.segment(3, 3) = torques_worldFrame_.e();
            robot_->setGeneralizedForce(genForces_);

            /// this option will visualize the applied forces and torques
            // robot_->setExternalForce(0, forces_worldFrame_);
            // robot_->setExternalTorque(0, torques_worldFrame_);
        }


        /********* Custom methods **********/
        void setTarget(double x, double y, double z) {
            targetPoint_[0] = x;
            targetPoint_[1] = y;
            targetPoint_[2] = z;
            if (visualizable_) visPoint->setPosition(targetPoint_.head(3));
        }

        void setRandomTargets(double radius, int updateRate) {
            if (updateTarget) {
                for (int i = 0; i < 3; i++) targetPoint_(i) = normDist_(gen_);
                targetPoint_.head(3) /= targetPoint_.head(3).norm();
                targetPoint_.head(3) *= radius; // target point has distance of 10 m within a sphere
                if (visualizable_) visPoint->setPosition(targetPoint_.head(3));
                updateTarget = false;
            }

            /// update target after specific amount of environment resets. updateRate is count in reversed order.
            if (loopCount_ <= 0) {
                loopCount_ = updateRate;
                updateTarget = true;
            }
            loopCount_--;
        }

        void setRandomStates(double pos, bool rot_bool, double vel, double angVel) {
            for (int i = 0; i < 3; i++) {
                gc_init_(i) = pos * normDist_(gen_);
                gv_init_(i) = vel * normDist_(gen_);
                gv_init_(i + 3) = angVel * normDist_(gen_);
            }
            if (rot_bool) {
                for (int i = 0; i < 4; i++) {
                    gc_init_(i + 3) = normDist_(gen_);
                }
                gc_init_.segment(3, 4) /= gc_init_.segment(3, 4).norm();
            }
        }

        void calculateEulerAngles() {
            double sinr_cosp = 2 * (quat_[0] * quat_[1] + quat_[2] * quat_[3]);
            double cosr_cosp = 1 - 2 * (quat_[1] * quat_[1] + quat_[2] * quat_[2]);
            eulerAngles_[0] = std::atan2(sinr_cosp, cosr_cosp);

            // pitch (y-axis rotation)
            double sinp = 2 * (quat_[0] * quat_[2] - quat_[3] * quat_[1]);
            if (std::abs(sinp) >= 1)
                eulerAngles_[1] = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
                else
                    eulerAngles_[1] = std::asin(sinp);

                // yaw (z-axis rotation)
                double siny_cosp = 2 * (quat_[0] * quat_[3] + quat_[1] * quat_[2]);
                double cosy_cosp = 1 - 2 * (quat_[2] * quat_[2] + quat_[3] * quat_[3]);
                eulerAngles_[2] = std::atan2(siny_cosp, cosy_cosp);

                for (int i; i < 3; i++) {
                    eulerAngles_[i] = std::abs(eulerAngles_[i]) + 1e-4;
                }
        }

        /// environment and quadcopter related variables
        bool visualizable_ = true;
        raisim::ArticulatedSystem *robot_;
        raisim::Visuals *visPoint;

        int gcDim_, gvDim_, nRotors_;
        Eigen::VectorXd gc_init_, gv_init_, gc_, gv_;
        Eigen::VectorXd obDouble_, targetPoint_;
        Eigen::Vector4d actionMean_, actionStd_;

        raisim::Mat<3, 3> worldRot_;
        raisim::Vec<4> quat_;
        Eigen::Vector3d bodyPos_, bodyLinVel_, bodyAngVel_;
        Eigen::Matrix3d bodyRot_;

        Eigen::Vector4d normedControlThrusts_;
        Eigen::VectorXd thrusts_, controlThrusts_;
        Eigen::Matrix4d thrusts2TorquesAndForces_;
        Eigen::Vector4d torquesAndForces_;
        Eigen::Vector3d torques_baseFrame_, forces_baseFrame_;
        raisim::Vec<3> torques_worldFrame_, forces_worldFrame_;
        Eigen::VectorXd genForces_;
        Eigen::Vector3d eulerAngles_;

        /// quadcopter model parameters
        const double rotorPos_ = 0.17104913036744201, momConst_ = 0.016, motConst = 20, thrConst = 8.54858e-06;
        const double dragCoeff = 8.06428e-05;
        const double rps_ = 2 * M_PI, rpm_ = rps_ / 60;
        const double g_ = 9.81, m_ = 1.727;
        double hoverThrust_ = m_ * g_ / 4;

        /// reward related variables
        raisim::Reward rewards_;
        std::set<size_t> bodyIndices_;
        double terminalRewardCoeff_ = -15.;
        double relativeAbsPosition;
        double relPositionReward;

        /// other variables
        bool updateTarget = false;
        int loopCount_ = 0;

        std::normal_distribution<double> normDist_;
        std::random_device rd_;
        std::mt19937 gen_;
    };
}

