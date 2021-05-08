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
   * @param headers Header files needed by the package.
   * @param sources Source files of the package.
   */
  Package(const std::string& name, const std::list<File*>& headers =
      std::list<File*>(), const std::list<File*>& sources =
      std::list<File*>());

  /**
   * Add package dependency.
   */
  void addPackage(const std::string& name);

  /**
   * Add header file.
   */
  void addHeader(const std::string& path);

  /**
   * Add source file.
   */
  void addSource(const std::string& path);

  Package* accept(Modifier* visitor);
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
   * Header files needed by the package.
   */
  std::list<File*> headers;

  /**
   * Source files of the package.
   */
  std::list<File*> sources;

  /**
   * All files (headers and sources).
   */
  std::list<File*> files;
};
}
