/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Braced.hpp"
#include "bi/common/Signature.hpp"
#include "bi/common/Scoped.hpp"

namespace bi {
/**
 * Type conversion function.
 *
 * @ingroup compiler_expression
 */
class ConversionParameter: public Expression, public Scoped, public Braced {
public:
  /**
   * Constructor.
   *
   * @param type Return type.
   * @param braces Braces expression.
   * @param loc Location.
   */
  ConversionParameter(Type* type, Expression* braces,
      shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~ConversionParameter();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Expression::definitely;
  using Expression::possibly;

  virtual bool dispatchDefinitely(const Expression& o) const;
  virtual bool definitely(const ConversionParameter& o) const;

  virtual bool dispatchPossibly(const Expression& o) const;
  virtual bool possibly(const ConversionParameter& o) const;
};
}
