/**
 * @file
 */
#pragma once

#include "bi/common/Scoped.hpp"
#include "bi/statement/File.hpp"

namespace bi {
/**
 * A package is a collection of files.
 */
class Package: public Scoped {
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
   * Destructor.
   */
  virtual ~Package();

  /**
   * Add header file.
   */
  void addHeader(const std::string& path);

  /**
   * Add source file.
   */
  void addSource(const std::string& path);

  Package* accept(Cloner* visitor) const;
  Package* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /**
   * Package name.
   */
  std::string name;

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
