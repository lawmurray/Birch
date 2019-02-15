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
   * Performance-tune the package.
   */
  void tune();

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
   */
  Package* createPackage();

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
   * @name Command-line options
   */
  //@{
  /**
   * Working directory.
   */
  fs::path work_dir;

  /**
   * Share directories.
   */
  std::list<fs::path> share_dirs;

  /**
   * Include directories.
   */
  std::list<fs::path> include_dirs;

  /**
   * Library directories.
   */
  std::list<fs::path> lib_dirs;

  /**
   * Target architecture.
   */
  std::string arch;

  /**
   * Installation directory.
   */
  std::string prefix;

  /**
   * Name of the package.
   */
  std::string packageName;

  /**
   * Description of the package.
   */
  std::string packageDesc;

  /**
   * Use unity build?
   */
  bool unity;

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
   * Are compiler warnings enabled?
   */
  bool warnings;

  /**
   * Is debugging mode enabled?
   */
  bool debug;

  /**
   * Is verbose reporting enabled?
   */
  bool verbose;

  /**
   * Is lazy deep clone enabled?
   */
  bool lazyDeepClone;

  /**
   * Is clone memoization enabled?
   */
  bool cloneMemo;

  /**
   * Is ancestry memoization enabled?
   */
  bool ancestryMemo;

  /**
   * Initial allocation size (number of entries) in maps used for clone
   * memoization.
   */
  int cloneMemoInitialSize;

  /**
   * Number of clone generations between deep clone memoizations.
   */
  int cloneMemoDelta;

  /**
   * Initial allocation size (number of entries) in sets used for ancestry
   * memoization.
   */
  int ancestryMemoInitialSize;

  /**
   * Number of clone generations between ancestry memoizations.
   */
  int ancestryMemoDelta;
  //@}

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
