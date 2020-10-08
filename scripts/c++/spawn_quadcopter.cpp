#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"

int main(int argc, char* argv[]) {
  auto binaryPath = raisim::Path::setFromArgv(argv[0]);
  raisim::World::setActivationKey( binaryPath.getDirectory() + "\\rsc\\activation.raisim");
  raisim::World world;
  auto quadcopter = world.addArticulatedSystem(binaryPath.getDirectory() + "\\rsc\\quadcopter\\Hummingbird\\urdf\\hummingbird.urdf");
  auto ground = world.addGround(-1);
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
