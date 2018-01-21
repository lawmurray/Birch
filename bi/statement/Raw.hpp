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
  Raw(Name* name, const std::string& raw,
      Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Raw();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /**
   * Raw C++ code.
   */
  std::string raw;
};
}
