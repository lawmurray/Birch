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
   * Run `ldconfig`, if on a platform where this is necessary, and running
   * under a user account where this is possible.
   */
  void ldconfig();

  /**
   * When a command fails to execute, get an explanation.
   */
  const char* explain(const std::string& cmd);

  /**
   * Compile the package and all dependencies with the current options, then
   * execute the `run` program of the package and return its total execution
   * time.
   */
  double time();

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
   * Is the memory pool enabled?
   */
  bool memoryPool;

  /**
   * Is lazy deep clone enabled?
   */
  bool lazyDeepClone;

  /**
   * Is clone memoization enabled?
   */
  bool cloneMemo;

  /**
   * Initial allocation size (number of entries) in maps used for clone
   * memoization.
   */
  int cloneMemoInitialSize;
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

  /**
   * Compile and run the package for each of a set of parameter values,
   * set the parameter to the value corresponding to the fastest time, and
   * return that time.
   *
   * @param parameter Pointer to the parameter to set (typically a member
   * variable of the same object, e.g. cloneMemoInitialSize).
   * @param values List of values of the parameter to try.
   *
   * @return The fastest time.
   */
  template<class T>
  double choose(T* parameter, const std::initializer_list<T>& values);
};
}

template<class T>
double bi::Driver::choose(T* parameter,
    const std::initializer_list<T>& values) {
  assert(values.size() > 0);

  double t = std::numeric_limits<double>::infinity();
  double best = t;
  int strikes = 0;
  T chosen = *values.begin();

  for (auto value = values.begin(); value != values.end() && strikes < 3;
      ++value) {
    *parameter = *value;
    std::cerr << '@' << *value << " = ";
    t = time();
    std::cerr << t << 's';
    if (t < best) {
      std::cerr << '*';
      best = t;
      chosen = *value;
      strikes = 0;
    } else {
      ++strikes;
    }
    std::cerr << std::endl;
  }
  *parameter = chosen;
  return t;
}
