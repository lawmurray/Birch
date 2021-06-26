/**
 * @file
 */
#pragma once

#include "src/primitive/system.hpp"
#include "src/statement/Package.hpp"

namespace birch {
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
   * Bootstrap package.
   */
  void bootstrap();

  /**
   * Configure package.
   */
  void configure();

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
   * Audit an existing package for possible issues.
   */
  void audit();

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
   * Transpile Birch files to C++.
   */
  void transpile();

  /**
   * Run make with a given target.
   *
   * @param cmd The target, empty string for the default target.
   */
  void target(const std::string& cmd = "");

  /**
   * Create the package.
   */
  Package* createPackage();

  /**
   * Consume a list of files from the build configuration file contents.
   *
   * @param key The key.
   */
  void readFiles(const std::string& key);

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
   * Installation prefix.
   */
  fs::path prefix;

  /**
   * Target architecture (native or empty).
   */
  std::string arch;

  /**
   * Compilation unit ("unity", "dir", or "file").
   */
  std::string unit;

  /**
   * Run mode ("debug", "test", or "release").
   */
  std::string mode;

  /**
   * Number of jobs for parallel build. If zero, a reasonable value is
   * determined from the environment.
   */
  int jobs;

  /**
   * Enable test build?
   */
  bool test;

  /**
   * Enable debug build?
   */
  bool debug;

  /**
   * Enable release build?
   */
  bool release;

  /**
   * Enable static library?
   */
  bool staticLib;

  /**
   * Enable shared library?
   */
  bool sharedLib;

  /**
   * Enable OpenMP?
   */
  bool openmp;

  /**
   * Enable compiler warnings?
   */
  bool warnings;

  /**
   * Enable compiler notes?
   */
  bool notes;

  /**
   * Enable translation of C++ compiler messages?
   */
  bool translate;

  /**
   * Enable verbose reporting?
   */
  bool verbose;

  /**
   * Is the bootstrap file new?
   */
  bool newBootstrap;

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
  std::map<std::string,std::list<std::string>> metaContents;
  std::map<std::string,std::list<fs::path>> metaFiles;
  std::set<fs::path> allFiles;

  /**
   * Leftover command-line arguments for program calls.
   */
  std::vector<char*> largv;
};
}
