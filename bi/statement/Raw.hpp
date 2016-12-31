/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/common/Named.hpp"

#include <string>

namespace bi {
/**
 * Raw C++ block.
 *
 * @ingroup compiler_statement
 */
class Raw: public Statement, public Named {
public:
  /**
   * Constructor.
   *
   * @param name Name (e.g. "cpp", "hpp").
   * @param raw Raw C++ code.
   * @param loc Location.
   */
  Raw(shared_ptr<Name> name, const std::string& raw,
      shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Raw();

  virtual Statement* acceptClone(Cloner* visitor) const;
  virtual Statement* acceptModify(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool operator<=(Statement& o);
  virtual bool operator==(const Statement& o) const;

  /**
   * Raw C++ code.
   */
  std::string raw;
};
}
