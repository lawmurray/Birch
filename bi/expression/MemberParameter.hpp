/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Numbered.hpp"
#include "bi/common/Valued.hpp"

namespace bi {
/**
 * Parameter to a class constructor.
 *
 * @ingroup compiler_expression
 */
class MemberParameter: public Expression,
    public Named,
    public Numbered,
    public Valued {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param type Type.
   * @param value Default value.
   * @param loc Location.
   */
  MemberParameter(Name* name, Type* type, Expression* value, Location* loc =
      nullptr);

  /**
   * Destructor.
   */
  virtual ~MemberParameter();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
