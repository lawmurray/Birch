/**
 * @file
 */
#pragma once

#include "src/statement/File.hpp"

namespace birch {
/**
 * A package is a collection of files.
 *
 * @ingroup statement
 */
class Package {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param sources Source files of the package.
   */
  Package(const std::string& name, const std::list<File*>& sources =
      std::list<File*>());

  /**
   * Add package dependency.
   */
  void addPackage(const std::string& name);

  /**
   * Add source file.
   */
  void addSource(const std::string& path);

  virtual void accept(Visitor* visitor) const;

  /**
   * Package name.
   */
  std::string name;

  /**
   * Package dependencies;
   */
  std::list<std::string> packages;

  /**
   * Source files of the package.
   */
  std::list<File*> sources;
};
}
