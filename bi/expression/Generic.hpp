/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Numbered.hpp"

namespace bi {
/**
 * Generic type.
 *
 * @ingroup expression
 */
class Generic: public Expression, public Named, public Numbered {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param type Type.
   * @param loc Location.
   */
  Generic(Name* name, Type* type, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Generic();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
