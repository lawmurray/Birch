/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/statement/File.hpp"
#include "bi/common/Path.hpp"
#include "bi/primitive/shared_ptr.hpp"

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
  Import(shared_ptr<Path> path, File* file, shared_ptr<Location> loc = nullptr);

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
  shared_ptr<Path> path;

  /**
   * File.
   */
  File* file;

  virtual possibly dispatch(Statement& o);
  virtual possibly le(Import& o);
};
}
