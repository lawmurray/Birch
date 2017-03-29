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
 * Function parameter.
 *
 * @ingroup compiler_expression
 */
class FuncParameter: public Expression,
    public Signature,
    public Scoped,
    public Braced {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param parens Parentheses expression.
   * @param result Result expression.
   * @param braces Braces expression.
   * @param form Function form.
   * @param loc Location.
   */
  FuncParameter(shared_ptr<Name> name, Expression* parens, Expression* result,
      Expression* braces, const SignatureForm form, shared_ptr<Location> loc =
          nullptr);

  /**
   * Constructor for binary operator.
   *
   * @param left Left operand.
   * @param name Operator.
   * @param right Right operand.
   * @param result Result expression.
   * @param braces Braces expression.
   * @param form Signature form.
   * @param loc Location.
   */
  FuncParameter(Expression* left, shared_ptr<Name> name, Expression* right,
      Expression* result, Expression* braces, const SignatureForm form,
      shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~FuncParameter();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Expression::definitely;
  using Expression::possibly;

  virtual bool dispatchDefinitely(Expression& o);
  virtual bool definitely(FuncParameter& o);

  virtual bool dispatchPossibly(Expression& o);
  virtual bool possibly(FuncParameter& o);
};
}
