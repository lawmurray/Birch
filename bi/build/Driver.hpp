/**
 * @file
 */
#pragma once

#include "bi/statement/Package.hpp"

#include "boost/filesystem.hpp"
#include "boost/interprocess/sync/file_lock.hpp"

#include <list>

namespace bi {
/**
 * Birch driver.
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
   * Destructor.
   */
  ~Driver();

  /**
   * Run user program.
   */
  void run(const std::string& prog);

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
   * Unlock the build directory.
   */
  void unlock();

private:
  /**
   * Read in the MANIFEST file.
   */
  void manifest();

  /**
   * Set up build directory.
   */
  void setup();

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
   * Lock the build directory.
   */
  void lock();

  /**
   * @name Command-line options
   */
  //@{
  /**
   * Working directory.
   */
  boost::filesystem::path work_dir;

  /**
   * Build directory.
   */
  boost::filesystem::path build_dir;

  /**
   * Share directories.
   */
  std::list<boost::filesystem::path> share_dirs;

  /**
   * Include directories.
   */
  std::list<boost::filesystem::path> include_dirs;

  /**
   * Library directories.
   */
  std::list<boost::filesystem::path> lib_dirs;

  /**
   * Installation directory.
   */
  boost::filesystem::path prefix;

  /**
   * Enable compiler warnings.
   */
  bool warnings;

  /**
   * Enable debugging mode.
   */
  bool debug;

  /**
   * Verbose reporting.
   */
  bool verbose;
  //@}

  /**
   * Name of the package.
   */
  std::string package_name;

  /**
   * The package.
   */
  Package* package;

  /**
   * Is the autogen.sh file new?
   */
  bool newAutogen;

  /**
   * Is the configure.ac file new?
   */
  bool newConfigure;

  /**
   * Is the common.am file new?
   */
  bool newMake;

  /**
   * Is MANIFEST new?
   */
  bool newManifest;

  /**
   * Is build directory locked?
   */
  bool isLocked;

  /**
   * File lock.
   */
  boost::interprocess::file_lock lockFile;

  /**
   * Lists of files from MANIFEST.
   */
  std::list<boost::filesystem::path> files, biFiles, cppFiles, hppFiles,
      metaFiles, otherFiles;

  /**
   * Leftover command-line arguments for program calls.
   */
  std::vector<char*> largv;
};
}
