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
#include "bi/dispatcher/Dispatcher.hpp"

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
   */
  FuncReference(shared_ptr<Name> name, Expression* parens,
      const SignatureForm form, shared_ptr<Location> loc = nullptr,
      FuncParameter* target = nullptr);

  /**
   * Constructor for binary operator.
   *
   * @param left Left operand.
   * @param name Operator.
   * @param right Right operand.
   * @param form Signature form.
   * @param loc Location.
   * @param target Target.
   */
  FuncReference(Expression* left, shared_ptr<Name> name, Expression* right,
      const SignatureForm form, shared_ptr<Location> loc = nullptr,
      FuncParameter* target = nullptr);

  /**
   * Destructor.
   */
  virtual ~FuncReference();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Expression::definitely;
  using Expression::possibly;

  virtual bool dispatchDefinitely(const Expression& o) const;
  virtual bool definitely(const FuncReference& o) const;
  virtual bool definitely(const FuncParameter& o) const;
  virtual bool definitely(const VarParameter& o) const;

  virtual bool dispatchPossibly(const Expression& o) const;
  virtual bool possibly(const FuncReference& o) const;
  virtual bool possibly(const FuncParameter& o) const;
  virtual bool possibly(const VarParameter& o) const;

  /**
   * Possible function resolutions that will need to be checked at runtime.
   */
  std::list<FuncParameter*> possibles;

  /**
   * Dispatcher to be used for this function, or `nullptr` if no dispatcher
   * is required.
   */
  Dispatcher* dispatcher;
};
}
