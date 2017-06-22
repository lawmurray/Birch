/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/expression/EmptyExpression.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Numbered.hpp"
#include "bi/common/Valued.hpp"
#include "bi/primitive/unique_ptr.hpp"

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
  MemberParameter(shared_ptr<Name> name, Type* type, Expression* value =
      new EmptyExpression(), shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~MemberParameter();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Expression::definitely;
  using Expression::possibly;

  virtual bool dispatchDefinitely(const Expression& o) const;
  virtual bool definitely(const MemberParameter& o) const;

  virtual bool dispatchPossibly(const Expression& o) const;
  virtual bool possibly(const MemberParameter& o) const;
};
}
