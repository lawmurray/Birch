/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/expression/FuncParameter.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Parenthesised.hpp"
#include "bi/common/Formed.hpp"
#include "bi/common/Reference.hpp"

#include <list>

namespace bi {
/**
 * Reference to function.
 *
 * @ingroup compiler_expression
 */
class FuncReference: public Expression,
    public Named,
    public Parenthesised,
    public Formed,
    public Reference<FuncParameter> {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param parens Expression in parentheses.
   * @param form Signature form.
   * @param loc Location.
   * @param target Target.
   * @param dispatcher Dispatcher.
   */
  FuncReference(shared_ptr<Name> name, Expression* parens,
      const SignatureForm form, shared_ptr<Location> loc = nullptr,
      FuncParameter* target = nullptr, Dispatcher* dispatcher = nullptr);

  /**
   * Constructor for binary operator.
   *
   * @param left Left operand.
   * @param name Operator.
   * @param right Right operand.
   * @param form Signature form.
   * @param loc Location.
   * @param target Target.
   * @param dispatcher Dispatcher.
   */
  FuncReference(Expression* left, shared_ptr<Name> name, Expression* right,
      const SignatureForm form, shared_ptr<Location> loc = nullptr,
      FuncParameter* target = nullptr, Dispatcher* dispatcher = nullptr);

  /**
   * Destructor.
   */
  virtual ~FuncReference();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Expression::definitely;
  using Expression::possibly;

  virtual bool dispatchDefinitely(Expression& o);
  virtual bool definitely(FuncReference& o);
  virtual bool definitely(FuncParameter& o);
  virtual bool definitely(Dispatcher& o);
  virtual bool definitely(VarParameter& o);

  virtual bool dispatchPossibly(Expression& o);
  virtual bool possibly(FuncReference& o);
  virtual bool possibly(FuncParameter& o);
  virtual bool possibly(Dispatcher& o);
  virtual bool possibly(VarParameter& o);

  /**
   * Runtime dispatcher.
   */
  Dispatcher* dispatcher;
};
}
