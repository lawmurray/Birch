/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/expression/Expression.hpp"
#include "bi/common/Braced.hpp"
#include "bi/common/Signature.hpp"
#include "bi/common/Typed.hpp"
#include "bi/common/Scoped.hpp"

namespace bi {
/**
 * Function parameter.
 *
 * @ingroup compiler_expression
 */
class FuncParameter: public Statement,
    public Signature,
    public Typed,
    public Scoped,
    public Braced {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param parens Parentheses expression.
   * @param type Typed type.
   * @param braces Braces expression.
   * @param form Function form.
   * @param loc Location.
   */
  FuncParameter(shared_ptr<Name> name, Expression* parens, Type* type,
      Expression* braces, const FunctionForm form, shared_ptr<Location> loc =
          nullptr);

  /**
   * Destructor.
   */
  virtual ~FuncParameter();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Statement::definitely;
  using Statement::possibly;

  virtual bool dispatchDefinitely(const Statement& o) const;
  virtual bool definitely(const FuncParameter& o) const;

  virtual bool dispatchPossibly(const Statement& o) const;
  virtual bool possibly(const FuncParameter& o) const;
};
}
