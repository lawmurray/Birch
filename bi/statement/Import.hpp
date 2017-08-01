/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/statement/File.hpp"
#include "bi/common/Path.hpp"

namespace bi {
/**
 * Import statement.
 *
 * @ingroup compiler_statement
 */
class Import: public Statement {
public:
  /**
   * Constructor.
   *
   * @param path Path.
   * @param file File associated with the path.
   * @param loc Location.
   */
  Import(Path* path, File* file, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Import();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /**
   * Path.
   */
  Path* path;

  /**
   * File.
   */
  File* file;
};
}
