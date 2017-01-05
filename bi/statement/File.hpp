/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/statement/EmptyStatement.hpp"
#include "bi/common/Scoped.hpp"
#include "bi/primitive/unique_ptr.hpp"

namespace bi {
/**
 * File.
 *
 * @ingroup File
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
   * @param import Import statements at top of file.
   * @param root Root statement of file (after imports).
   */
  File(const std::string& path, Statement* imports = new EmptyStatement(),
      Statement* root = new EmptyStatement());

  /**
   * Destructor.
   */
  virtual ~File();

  File* accept(Cloner* visitor) const;
  void accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /**
   * File name.
   */
  std::string path;

  /**
   * Import statements.
   */
  unique_ptr<Statement> imports;

  /**
   * Root statement of file (after imports).
   */
  unique_ptr<Statement> root;

  /**
   * Parsing state of this file.
   */
  State state;
};
}
