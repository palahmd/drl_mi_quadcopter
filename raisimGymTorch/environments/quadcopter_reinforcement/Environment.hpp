//
// Created by pala on 17.11.20.
//
#pragma once

#include <stdlib.h>
#include <random>
#include <cstdint>
#include <set>
#include "../../RaisimGymEnv.hpp"
#include "include/quaternionToEuler.hpp"
#include <cmath>
#include "Eigen/Dense"


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

            /// get robot data
            gcDim_ = robot_->getGeneralizedCoordinateDim();
            gvDim_ = robot_->getDOF();
            nRotors_ = gvDim_ - 6;
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

            /// nominal configuration of quadcopter: [0]-[2]: center of mass, [3]-[6]: quanternions, [7]-[10]: rotors
            gc_init_ << 0.0, 0.0, 0.135, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
            gv_init_ << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1000 * rpm_, 1000 * rpm_, -1000 * rpm_, 1000 * rpm_;

            /// initialize rotor thrusts_ and conversion matrix for generated forces and torques
            thrusts_.setZero(nRotors_);
            controlThrusts_.setZero(nRotors_);
            thrusts2TorquesAndForces_ << 1, 1, 1, 1,
                    rotorPos_, -rotorPos_, -rotorPos_, rotorPos_,
                    -rotorPos_, -rotorPos_, rotorPos_, rotorPos_,
                    momConst_, -momConst_, momConst_, -momConst_;

            /// action & observation scaling
            actionMean_.setConstant(hoverThrust_);
            actionStd_.setConstant(0.5*hoverThrust_);

            /// indices of links that should not make contact with ground - all links and bodies
            bodyIndices_.insert(robot_->getBodyIdx("rotor_0"));
            bodyIndices_.insert(robot_->getBodyIdx("rotor_1"));
            bodyIndices_.insert(robot_->getBodyIdx("rotor_2"));
            bodyIndices_.insert(robot_->getBodyIdx("rotor_3"));

            /// Reward coefficients
            rewards_.initializeFromConfigurationFile (cfg["reward"]);

            /// visualize if it is the first environment
            if (visualizable_) {
                server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                server_->launchServer();
                server_->focusOn(robot_);
                visPoint = server_->addVisualSphere("visPoint", 0.25, 0.8, 0, 0);
            }
        }

        void init() final {}

        void reset() final {
            robot_->setState(gc_init_, gv_init_);

            /// set random target point
            targetPoint_[0] = generateRandomValue(-5, 5);
            targetPoint_[1] = generateRandomValue(-5, 5);
            targetPoint_[2] = generateRandomValue(2, 5);
            if (visualizable_){
                server_->focusOn(robot_);
                visPoint->setPosition(targetPoint_.head(3));
            }

            updateObservation();
        }

        float step(const Eigen::Ref<EigenVec> &action) final {
            controlThrusts_ = action.cast<double>();

            double max_scale = controlThrusts_.maxCoeff();
            double min_scale = controlThrusts_.minCoeff();

            /// scale action down to [-1, 1]. should work better than clipping, does at least for the pid controller
            if ((max_scale > 1) || (min_scale < - 1)) {
                if (std::abs(min_scale) < std::abs(max_scale)) {
                    controlThrusts_ /= max_scale;
                }
                else {
                    controlThrusts_ /= min_scale;
                }
            }
            normedControlThrusts_ = controlThrusts_;

            /// scale bounded action input to thrusts
            controlThrusts_ = controlThrusts_.cwiseProduct(actionStd_);
            controlThrusts_ += actionMean_;

            for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
                if (server_) server_->lockVisualizationServerMutex();
                applyThrusts();
                world_->integrate();
                /// apply simple motor delay model with rotor delay
                for (int i = 0; i<4; i++){
                    if (thrusts_[i]<controlThrusts_[i]) {  // time constant for increasing rotor speed
                        thrusts_[i] = thrusts_[i] + (controlThrusts_[i] - thrusts_[i]) * simulation_dt_ / 0.0125;
                    } else if (thrusts_[i]>controlThrusts_[i]){   // time constant for decreasing rotor speed
                        thrusts_[i] = thrusts_[i] + (controlThrusts_[i] - thrusts_[i]) * simulation_dt_ / 0.025;
                    }
                }
                if (server_) server_->unlockVisualizationServerMutex();
            }

            updateObservation();

            rewards_.record("position", std::sqrt((targetPoint_.head(3) - bodyPos_).transpose() * (targetPoint_.head(3) - bodyPos_)));
            rewards_.record("thrust", normedControlThrusts_.mean());
            rewards_.record("orientation", std::abs(eulerAngles_(2)));
            rewards_.record("angularVelocity", std::abs(bodyAngVel_.mean()));


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
            calculateEulerAngles();

            /// observation vector (later for RL-Algorithm)
            for (size_t i = 0; i < 3; i++) {
                obDouble_[i] = bodyPos_[i];
            }

            // World Frame rotation Matrix and quaternion: obDouble_[9]-obDouble_[17], ob_q[9]-ob_q[12]
            for (size_t i = 0; i < 3; i++) {
                for (size_t j = 0; j<3; j++){
                    obDouble_[j + (i+1)*3] = bodyRot_(i,j);
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

            for (size_t i = 0; i < 4; i++) {
                obDouble_[i + 18] = quat_.e()[i];
            }
        }

        double generateRandomValue(int MIN, int MAX){
            std::random_device rd;
            std::default_random_engine eng(rd());
            std::uniform_real_distribution<double> distr(MIN, MAX);

            return distr(eng);
        }

        void calculateEulerAngles(){
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
        }

        void observe(Eigen::Ref<EigenVec> ob) final {
            /// convert it to float
            obDouble_ -= targetPoint_;
            ob = obDouble_.cast<float>();
        }

        void applyThrusts(){
            /// calculate Forces and Torques
            torquesAndForces_ = thrusts2TorquesAndForces_ * thrusts_;
            forces_baseFrame_ << 0.0, 0.0, torquesAndForces_(0);
            torques_baseFrame_ = torquesAndForces_.tail(3);

            torques_worldFrame_.e() = worldRot_.e() * torques_baseFrame_;
            forces_worldFrame_.e() = worldRot_.e() * forces_baseFrame_;

            genForces_.head(3) = forces_worldFrame_.e();
            genForces_.segment(3,3) = torques_worldFrame_.e();
            robot_->setGeneralizedForce(genForces_);

            /// this will visualize the applied forces and torques
            // robot_->setExternalForce(0, forces_worldFrame_);
            // robot_->setExternalTorque(0, torques_worldFrame_);
        }
        bool isTerminalState(float& terminalReward) final {
            terminalReward = float(terminalRewardCoeff_);


            for(auto& contact: robot_->getContacts()) {
                if (bodyIndices_.find(contact.getlocalBodyIndex()) == bodyIndices_.end())
                    terminalReward = -10.f;
                    return true;
            }

            if (std::abs((gc_.head(3)).mean()) > 30){
                return true;
            }

            terminalReward = 0.f;
            return false;
        }

        raisim::Mat<3,3> worldRot_;
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
        const double rotorPos_ = 0.17104913036744201, momConst_ = 0.016;
        const double rps_ = 2 * M_PI, rpm_ = rps_/60;
        const double g_ = 9.81, m_ = 1.727;
        const double hoverThrust_ = m_ * g_ / 4;
        int loopCount_;

        int gcDim_, gvDim_, nRotors_;
        bool visualizable_ = true;
        raisim::ArticulatedSystem* robot_;
        Eigen::VectorXd gc_init_, gv_init_, gc_, gv_;
        double terminalRewardCoeff_ = -5.;
        Eigen::VectorXd obDouble_, targetPoint_;
        Eigen::Vector4d actionMean_, actionStd_;
        std::set<size_t> baseIndex_;
        raisim::Reward rewards_;
        std::set<size_t> bodyIndices_;

        raisim::Visuals* visPoint;
    };
}

