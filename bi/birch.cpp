/**
 * @file
 *
 * The driver program.
 */
#include "bi/build/Driver.hpp"
#include "bi/exception/DriverException.hpp"
#include "bi/build/misc.hpp"

#include <iostream>
#include <gc.h>

int main(int argc, char** argv) {
  using namespace boost::filesystem;
  using namespace bi;

  /* initialise garbage collector */
  GC_INIT();

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
      driver.build();
      driver.install();
    } else if (prog.compare("create") == 0) {
      driver.create();
    } else if (prog.compare("validate") == 0) {
      driver.validate();
    } else if (prog.compare("docs") == 0) {
      driver.docs();
    } else {
      driver.build();
      driver.install();
      driver.unlock();
      driver.run(prog + "_");  // underscore suffix for user-specified names
    }
  } catch (Exception& e) {
    std::cerr << e.msg << std::endl;
  }

  return 0;
}
