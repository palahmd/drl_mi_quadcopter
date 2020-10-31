#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"
#include "Eigen/Dense"


int main(int argc, char* argv[]) {
  auto binaryPath = raisim::Path::setFromArgv(argv[0]);
  raisim::World::setActivationKey( binaryPath.getDirectory() + "\\rsc\\activation.raisim");

  raisim::World world;
  world.setTimeStep(0.002);

  auto ground = world.addGround();
  auto quadcopter = world.addArticulatedSystem(binaryPath.getDirectory() + "\\rsc\\quadcopter\\ITM_Quadrocopter\\urdf\\ITM_Quadrocopter.urdf");

  int gcDim_ = quadcopter->getGeneralizedCoordinateDim();
  int gvDim_ = quadcopter->getDOF();
  int nRotors_ = gvDim_ - 6;

  /// initialize containers
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
  gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
  gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
  pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nRotors_);

  /// nominal configuration of quadcopter: [0]-[2]: center of mass, [3]-[6]: quanternions, [7]-[10]: rotors
  gc_init_ << 0, 0, 0.1433, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

  /// for rotor movement simulation. Rotors have no mass
  /*
  quadcopter->setMass(1, 0);
  quadcopter->setMass(2, 0);
  quadcopter->setMass(3, 0);
  quadcopter->setMass(4,0);
  */
  gv_init_ << 0, 0, 10, 0, 10, 0, -838, 838, 838, -838;

  quadcopter->setName("Quaddy");
  quadcopter->setState(gc_init_, gv_init_);
  quadcopter->setControlMode(raisim::ControlMode::FORCE_AND_TORQUE);
  quadcopter->setGeneralizedForce({0, 0, 0, 0, 0, 0, 0, 0, 0, 0});


  /// launch raisim server for visualization. Can be visualized on raisimUnity
  raisim::RaisimServer server(&world);
  server.launchServer();
  server.focusOn(quadcopter);

    for (int i=0; i<20000; i++){
    raisim::MSLEEP(2);
    server.integrateWorldThreadSafe();
    
  }

  server.killServer();
}
