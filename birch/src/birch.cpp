/**
 * @file
 *
 * The driver program.
 */
#include "src/build/Driver.hpp"
#include "src/primitive/system.hpp"

#include <string>
#include <cstdlib>
#include <csignal>
#include <dlfcn.h>

int main(int argc, char** argv) {
  try {
    /* first option (should be a program name) */
    std::string prog = argc > 1 ? argv[1]: "help";

    birch::Driver driver(argc - 1, argv + 1);
    if (prog.compare("bootstrap") == 0) {
      driver.bootstrap();
    } else if (prog.compare("configure") == 0) {
      driver.configure();
    } else if (prog.compare("build") == 0) {
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
    } else if (prog.compare("audit") == 0) {
      driver.audit();
    } else if (prog.compare("docs") == 0) {
      driver.docs();
    } else if (prog.compare("help") == 0 || prog.compare("--help") == 0) {
      driver.help();
    } else {
      void* handle;

      /* dynamically load the NumBirch backend */
      handle = dlopen(driver.numbirch().c_str(), RTLD_NOW|RTLD_GLOBAL);
      if (!handle) {
        std::cerr << dlerror() << std::endl;
      }

      /* dynamically load the shared library for the package, which will
       * populate programs, then try to find the named program */
      handle = dlopen(driver.library().c_str(), RTLD_NOW|RTLD_GLOBAL);
      if (handle) {
        void* sym = dlsym(handle, "retrieve_program");
        if (sym) {
          typedef int prog_t(int argc, char** argv);
          typedef prog_t* retrieve_program_t(const std::string&);
          auto retrieve_program = reinterpret_cast<retrieve_program_t*>(sym);
          prog_t* f = retrieve_program(prog);
          if (f) {
            return f(driver.argc(), driver.argv());
          } else {
            std::cerr << "no program " << prog << std::endl;
          }
        } else {
          std::cerr << "no symbol retrieve_program; standard library broken?"
              << std::endl;
        }
      } else {
        std::cerr << dlerror() << std::endl;
      }
    }
  } catch (const birch::Exception& e) {
    std::cerr << e.msg << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
