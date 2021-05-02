/**
 * @file
 */
#pragma once

#include "src/statement/EmptyStatement.hpp"

namespace birch {
/**
 * File.
 *
 * @ingroup statement
 */
class File {
public:
  /**
   * Constructor.
   *
   * @param path File path.
   * @param root Root statement of file.
   */
  File(const std::string& path, Statement* root = new EmptyStatement());

  File* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /**
   * File name.
   */
  std::string path;

  /**
   * Root statement of file.
   */
  Statement* root;
};
}
