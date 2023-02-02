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
#ifdef HAVE_EXECINFO_H
#include <execinfo.h>
#endif

/*
 * Signal handler.
 */
static void abort(int sig) {
  #ifdef HAVE_EXECINFO_H
  static const int maxSize = 20;
	void* trace[maxSize];
	int size = 0;
	char **messages = nullptr;

  std::regex parse("(.*?)(?:\\((.*?)\\))? \\[0x[0-9a-f]+\\]");
  std::regex ext("(\\.birch:\\d+)(?::\\d+)?$");
  std::regex fname("(\\w+)(?:\\(.*\\))?$");
  
	size = backtrace(trace, maxSize);
	messages = backtrace_symbols(trace, size);

  /* now we attempt to map symbols to source file locations; here there are
   * two utils, addr2line and eu-addr2line, anecdotally the latter seems
   * better at looking up symbols and providing relative (rather than
   * absolute) file names, while the former seems better at demangling
   * function names; we use the two in combination */
  for (int i = 0; i < size; ++i) {
    std::cmatch m;
    std::regex_search(messages[i], m, parse);
    std::string file(m[1].str());
    std::string addr(m[2].str());

    if (file != "birch") {
      std::string demangled, source;
      std::stringstream cmd;
      char* line = nullptr;
      size_t n = 0;
      FILE* pipe = nullptr;
      int status = 0;

      /* backtrace_symbols() gives addresses as function+offset, and addr2line
      * seems to not find the source file location based on this; instead use
      * eu-addr2line first find an absolute hex address */
      cmd.str("");
      cmd << "eu-addr2line -a -e " << file << ' ' << addr << " 2> /dev/null";
      pipe = popen(cmd.str().c_str(), "r");
      if (pipe) {
        /* absolute address */
        if (getline(&line, &n, pipe) > 0) {
          addr = line;
          addr.pop_back(); // remove new line
          free(line);
          line = nullptr;
          n = 0;
        }

        /* source file location */
        if (getline(&line, &n, pipe) > 0) {
          source = line;
          source.pop_back(); // remove new line
          free(line);
          line = nullptr;
          n = 0;

          /* remove the extra number from the end of the source file location
          * (possibly the column, but inaccurate for Birch sources anyway),
          * filename:L:C -> filename:L */
          source = std::regex_replace(source, ext, "$1");
        }
        status = pclose(pipe);
      }

      /* now use the (possibly) updated address with addr2line to demangle the
       * function name */
      cmd.str("");
      cmd << "addr2line -C -f -e " << file << ' ' << addr << " 2> /dev/null";
      pipe = popen(cmd.str().c_str(), "r");
      if (pipe) {
        if (getline(&line, &n, pipe) > 0) {
          demangled = line;
          demangled.pop_back(); // remove new line
          free(line);
          line = nullptr;
          n = 0;
        }
        status = pclose(pipe);

        /* some function names are very complex, with namespaces, generic type
         * arguments, etc; if we can match the end of the name to a simple
         * "function(...)" then use that, otherwise replace with "..." to
         * avoid too much information */
        std::smatch m1;
        if (std::regex_search(demangled, m1, fname)) {
          demangled = m1[1].str();
        } else {
          demangled = "...";
        }
      }

      /* pretty-print the stack frame, only if it's from a *.birch file */
      if (std::regex_search(source, ext)) {
        fprintf(stderr, "    %-24s @ %s\n", demangled.c_str(), source.c_str());
      }
    }
  }
  free(messages);
  #endif
}

int main(int argc, char** argv) {
  /* set up signal handling */
  std::signal(SIGABRT, abort);

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
    } else if (prog.compare("abort") == 0) {
      abort(0);
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
