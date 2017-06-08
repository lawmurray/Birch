/**
 * @file
 *
 * The driver program.
 */
#include "bi/build/Driver.hpp"
#include "bi/build/Packager.hpp"
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

    if (prog.compare("build") == 0) {
      Driver driver(argc - 1, argv + 1);
      driver.build();
    } else if (prog.compare("install") == 0) {
      Driver driver(argc - 1, argv + 1);
      driver.build();
      driver.install();
    } else if (prog.compare("create") == 0) {
      Packager packager(argc - 1, argv + 1);
      packager.create();
    } else if (prog.compare("validate") == 0) {
      Packager packager(argc - 1, argv + 1);
      packager.validate();
    } else if (prog.compare("distribute") == 0) {
      Packager packager(argc - 1, argv + 1);
      packager.distribute();
    } else {
      Driver driver(argc - 1, argv + 1);
      driver.build();
      driver.install();
      driver.unlock();
      driver.run(prog);
    }
  } catch (Exception& e) {
    std::cerr << e.msg << std::endl;
  }

  return 0;
}
