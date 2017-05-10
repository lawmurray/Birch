/**
 * @file
 */
#pragma once

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
   * Run package.
   */
  void run(const std::string& prog);

  /**
   * Unlock the build directory.
   */
  void unlock();

private:
  /**
   * Set up build directory.
   */
  void setup();

  /**
   * Run autogen.sh.
   */
  void autogen();

  /**
   * Run configure.
   */
  void configure();

  /**
   * Run make.
   */
  void make();

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
   * Enable Birch standard library.
   */
  bool enable_std;

  /**
   * Enable compiler warnings.
   */
  bool enable_warnings;


  /**
   * Enable debugging mode.
   */
  bool enable_debug;

  /**
   * Do not build.
   */
  bool dry_build;

  /**
   * Do not run.
   */
  bool dry_run;

  /**
   * Force all build steps to be performed, even when determined not to be
   * required.
   */
  bool force;

  /**
   * Verbose reporting.
   */
  bool verbose;
  //@}

  /**
   * Name of the package.
   */
  std::string packageName;

  /**
   * Local command-line arguments.
   */
  std::vector<char*> largv;

  /**
   * Buffers for local command-line arguments.
   */
  std::list<std::string> fbufs;

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
   * File lock.
   */
  boost::interprocess::file_lock lockFile;

  /**
   * Is build directory locked?
   */
  bool isLocked;
};
}
