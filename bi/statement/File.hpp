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
 * @ingroup compiler_statement
 */
class File: public Scoped {
public:
  /**
   * States of file during compilation.
   */
  enum State {
    UNRESOLVED, RESOLVING, RESOLVED, UNGENERATED, GENERATING, GENERATED
  };

  /**
   * Constructor.
   *
   * @param path File path.
   * @param root Root statement of file.
   */
  File(const std::string& path = "", Statement* root = new EmptyStatement());

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

  /**
   * Parsing state of this file.
   */
  State state;
};
}
