/**
 * @file
 *
 * The driver program.
 */
#include "bi/build/Driver.hpp"
#include "bi/exception/DriverException.hpp"
#include "bi/build/misc.hpp"

#include <iostream>

int main(int argc, char** argv) {
  using namespace boost::filesystem;
  using namespace bi;

  try {
    /* first option (should be a program name) */
    std::string prog;
    if (argc > 1) {
      prog = argv[1];
    } else {
      throw DriverException("No command given.");
    }

    Driver driver(argc - 1, argv + 1);
    if (prog.compare("build") == 0) {
      driver.build();
    } else if (prog.compare("install") == 0) {
      driver.install();
    } else if (prog.compare("uninstall") == 0) {
      driver.uninstall();
    } else if (prog.compare("dist") == 0) {
      driver.dist();
    } else if (prog.compare("clean") == 0) {
      driver.clean();
    } else if (prog.compare("init") == 0) {
      driver.init();
    } else if (prog.compare("check") == 0) {
      driver.check();
    } else if (prog.compare("docs") == 0) {
      driver.docs();
    } else {
      driver.run(prog + "_");  // underscore suffix for user-specified names
    }
  } catch (Exception& e) {
    std::cerr << e.msg << std::endl;
  }

  return 0;
}
