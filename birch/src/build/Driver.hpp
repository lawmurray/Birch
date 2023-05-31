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
   * Remaining number of command-line options after processing those
   * recognized.
   */
  int argc();

  /**
   * Remaining command-line options after processing those recognized.
   */
  char** argv();

  /**
   * Determine name of the shared library for the NumBirch backend required.
   */
  std::string numbirch();

  /**
   * Determine name of the shared library for this package.
   */
  std::string library();

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
   * Read in the configuration file.
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
   * Floating point precision run mode ("single" or "double").
   */
  std::string precision;

  /**
   * NumBirch backend ("eigen" or "cuda").
   */
  std::string backend;

  /**
   * Number of jobs for parallel build. If zero, a reasonable value is
   * determined from the environment.
   */
  int jobs;

  /**
   * Enable single-precision builds?
   */
  bool enableSingle;

  /**
   * Enable double-precision builds?
   */
  bool enableDouble;

  /**
   * Enable static library?
   */
  bool enableStatic;

  /**
   * Enable shared library?
   */
  bool enableShared;

  /**
   * Enable standalone program build?
   */
  bool enableStandalone;

  /**
   * Enable OpenMP?
   */
  bool enableOpenmp;

  /**
   * Enable assertions?
   */
  bool enableAssert;

  /**
   * Enable optimizations?
   */
  bool enableOptimize;

  /**
   * Enable debug information?
   */
  bool enableDebug;

  /**
   * Enable coverage information?
   */
  bool enableCoverage;

  /**
   * Enable compiler warnings?
   */
  bool enableWarnings;

  /**
   * Enable compiler notes?
   */
  bool enableNotes;

  /**
   * Enable translation of compiler messages from C++ to Birch?
   */
  bool enableTranslate;

  /**
   * Enable verbose reporting?
   */
  bool enableVerbose;

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
