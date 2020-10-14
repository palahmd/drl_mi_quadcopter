#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"
#include "Eigen/Dense"

int main(int argc, char* argv[]) {
  auto binaryPath = raisim::Path::setFromArgv(argv[0]);
  raisim::World::setActivationKey( binaryPath.getDirectory() + "\\rsc\\activation.raisim");
  raisim::World world;
  auto quadcopter = world.addArticulatedSystem(binaryPath.getDirectory() + "\\rsc\\quadcopter\\Hummingbird\\urdf\\hummingbird.urdf");
  auto ground = world.addGround(-1);
  Eigen::VectorXd init_State;
  init_State << 0,0,50, 1, 0, 0, 0, 0, 0, 0;
  Eigen::VectorXd init_Vel;
  init_Vel << 0,0,10, 0, 0, 0, 0, 0, 0, 0;
  quadcopter->setState(init_State, init_Vel);
  world.setTimeStep(0.002);

  /// launch raisim server for visualization. Can be visualized on raisimUnity
  raisim::RaisimServer server(&world);
  server.launchServer();
  server.focusOn(quadcopter);

  while (1) {
    raisim::MSLEEP(2);
    server.integrateWorldThreadSafe();
    
  }

  server.killServer();
}
