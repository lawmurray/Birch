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

  virtual Statement* acceptClone(Cloner* visitor) const;
  virtual void acceptModify(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool operator<=(Statement& o);
  virtual bool operator==(const Statement& o) const;

  /**
   * Path.
   */
  shared_ptr<Path> path;

  /**
   * File.
   */
  File* file;
};
}
