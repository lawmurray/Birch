/**
 * @file
 */
#pragma once

#include "boost/filesystem.hpp"

#include <list>

namespace bi {
/**
 * Birch packager.
 *
 * @ingroup driver
 */
class Packager {
public:
  /**
   * Constructor.
   */
  Packager(int argc, char** argv);

  /**
   * Destructor.
   */
  ~Packager();

  /**
   * Create a new package.
   */
  void create();

  /**
   * Validate and existing package.
   */
  void validate();

  /**
   * Prepare for distribution.
   */
  void distribute();

private:
  /**
   * @name Command-line options
   */
  //@{
  /**
   * Share directories.
   */
  std::list<boost::filesystem::path> share_dirs;

  /**
   * Force all steps without prompting for e.g. overwriting of existing
   * files.
   */
  bool force;

  /**
   * Verbose reporting.
   */
  bool verbose;
  //@}
};
}
