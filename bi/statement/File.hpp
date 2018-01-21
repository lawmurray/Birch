/**
 * @file
 */
#pragma once

#include "bi/common/Scoped.hpp"
#include "bi/statement/EmptyStatement.hpp"

namespace bi {
/**
 * File.
 *
 * @ingroup statement
 */
class File: public Scoped {
public:
  /**
   * Constructor.
   *
   * @param path File path.
   * @param path Root scope.
   * @param root Root statement of file.
   */
  File(const std::string& path, Scope* scope, Statement* root =
      new EmptyStatement());

  /**
   * Destructor.
   */
  virtual ~File();

  File* accept(Cloner* visitor) const;
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
