/**
 * @file
 */
#pragma once

#include "bi/build/misc.hpp"
#include "bi/statement/Package.hpp"

namespace bi {
/**
 * Driver.
 *
 * @ingroup driver
 */
class Driver {
public:
  /**
   * Constructor.
   */
  Driver(int argc, char** argv);

  /**
   * Run user program.
   *
   * @param prog Name of the program.
   * @param xargs Extra arguments.
   */
  void run(const std::string& prog, const std::vector<char*>& xargs = {});

  /**
   * Build package.
   */
  void build();

  /**
   * Install package.
   */
  void install();

  /**
   * Uninstall package.
   */
  void uninstall();

  /**
   * Distribute package.
   */
  void dist();

  /**
   * Clean package.
   */
  void clean();

  /**
   * Create a new package.
   */
  void init();

  /**
   * Validate an existing package.
   */
  void check();

  /**
   * Produce documentation.
   */
  void docs();

  /**
   * Print the help message.
   */
  void help();

private:
  /**
   * Read in the META.json file.
   */
  void meta();

  /**
   * Set up build directory.
   */
  void setup();

  /**
   * Create the package.
   *
   * @param includeRequires Include checks for required packages?
   */
  Package* createPackage(bool includeRequires);

  /**
   * Compile Birch files to C++.
   */
  void compile();

  /**
   * Run autogen.sh.
   */
  void autogen();

  /**
   * Run configure.
   */
  void configure();

  /**
   * Run make with a given target.
   *
   * @param cmd The target, empty string for the default target.
   */
  void target(const std::string& cmd = "");

  /**
   * Run `ldconfig`, if on a platform where this is necessary, and running
   * under a user account where this is possible.
   */
  void ldconfig();

  /**
   * When a command fails to execute, get an explanation.
   */
  const char* explain(const std::string& cmd);

  /**
   * Produce a suffix to use on the build directory name, where this is
   * unique to the particular configuration.
   */
  std::string suffix() const;

  /**
   * Consume a list of files from the meta file.
   *
   * @param meta Property tree of the meta file.
   * @param key The key.
   * @param checkExists Check if the files exists?
   */
  void readFiles(const boost::property_tree::ptree& meta,
      const std::string& key, const bool checkExists);

  /**
   * Package name.
   */
  std::string packageName;

  /**
   * Package version.
   */
  std::string packageVersion;

  /**
   * Package description.
   */
  std::string packageDescription;

  /**
   * Working directory.
   */
  fs::path workDir;

  /**
   * Destination directory (`DESTDIR` given to `make install`).
   */
  std::string destDir;

  /**
   * Installation prefix.
   */
  fs::path prefix;

  /**
   * Installation directory for executables.
   */
  fs::path binDir;

  /**
   * Installation directory for libraries.
   */
  fs::path libDir;

  /**
   * Installation directory for headers.
   */
  fs::path includeDir;

  /**
   * Installation directory for data.
   */
  fs::path dataDir;

  /**
   * Target architecture (native, js, or wasm).
   */
  std::string arch;

  /**
   * Build mode (debug, test, or release).
   */
  std::string mode;

  /**
   * Compilation unit (unity, dir, or file).
   */
  std::string unit;

  /**
   * Build static library?
   */
  bool staticLib;

  /**
   * Build shared library?
   */
  bool sharedLib;

  /**
   * Is OpenMP enabled?
   */
  bool openmp;

  /**
   * Number of jobs for parallel build. If zero, a reasonable value is
   * determined from the environment.
   */
  int jobs;

  /**
   * Are compiler warnings enabled?
   */
  bool warnings;

  /**
   * Are compiler notes enabled?
   */
  bool notes;

  /**
   * Is verbose reporting enabled?
   */
  bool verbose;

  /**
   * Is the autogen.sh file new?
   */
  bool newAutogen;

  /**
   * Is the configure.ac file new?
   */
  bool newConfigure;

  /**
   * Is the Makefile.am file new?
   */
  bool newMake;

  /**
   * Share directories.
   */
  std::list<fs::path> shareDirs;

  /**
   * Include directories.
   */
  std::list<fs::path> includeDirs;

  /**
   * Library directories.
   */
  std::list<fs::path> libDirs;


  /**
   * Lists of files from meta.
   */
  std::map<std::string,std::list<fs::path>> metaFiles;
  std::set<fs::path> allFiles;

  /**
   * Leftover command-line arguments for program calls.
   */
  std::vector<char*> largv;
};
}
