/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Signature.hpp"
#include "bi/common/Scoped.hpp"
#include "bi/common/Parameter.hpp"
#include "bi/primitive/poset.hpp"
#include "bi/primitive/possibly.hpp"

namespace bi {
/**
 * Dispatcher for runtime resolution of a function call.
 *
 * @ingroup compiler_expression
 */
class Dispatcher: public Expression,
    public Signature,
    public Scoped,
    public Parameter<Expression> {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param parens Parentheses expression.
   * @param result Result expression.
   * @param loc Location.
   */
  Dispatcher(shared_ptr<Name> name, Expression* parens, Expression* result,
      shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Dispatcher();

  /**
   * Insert a function into this dispatcher. The mangled name of the function
   * must match the name of the dispatcher.
   */
  void insert(FuncParameter* func);

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /**
   * Functions handled by this dispatcher.
   */
  poset<FuncParameter*,bi::possibly> funcs;

  using Expression::definitely;
  using Expression::possibly;

  virtual bool dispatchDefinitely(Expression& o);
  virtual bool definitely(Dispatcher& o);

  virtual bool dispatchPossibly(Expression& o);
  virtual bool possibly(Dispatcher& o);
  virtual bool possibly(FuncParameter& o);

private:
  /**
   * Update the variant types of parameters.
   */
  void update(Expression* o1, Expression* o2);
};
}
