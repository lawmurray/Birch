/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/common/Named.hpp"

namespace bi {
/**
 * Basic (built-in) type.
 *
 * @ingroup compiler_type
 */
class Basic: public Statement, public Named {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param loc Location.
   */
  Basic(Name* name, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Basic();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
