/**
 * @file
 */
#pragma once

#include "bi/statement/File.hpp"

#include <list>

namespace bi {
/**
 * A package is a collection of files.
 */
class Package {
public:
  /**
   * Constructor.
   *
   * @param files List of files.
   */
  Package(const std::list<File*>& files);

  /**
   * Destructor.
   */
  virtual ~Package();

  Package* accept(Cloner* visitor) const;
  Package* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /**
   * Files contained in the package.
   */
  std::list<File*> files;
};
}
