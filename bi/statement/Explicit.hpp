/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/common/Based.hpp"

namespace bi {
/**
 * Declaration that an explicit instantiation of a generic class is available.
 *
 * @ingroup statement
 */
class Explicit: public Statement, public Based {
public:
  /**
   * Constructor.
   *
   * @param base Base type.
   * @param loc Location.
   */
  Explicit(Type* base, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Explicit();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
