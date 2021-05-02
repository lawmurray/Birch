/**
 * @file
 */
#pragma once

#include "src/statement/Statement.hpp"
#include "src/common/Named.hpp"

namespace birch {
/**
 * Raw C++ block.
 *
 * @ingroup statement
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
  Raw(Name* name, const std::string& raw, Location* loc = nullptr);

  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /**
   * Raw C++ code.
   */
  std::string raw;
};
}
