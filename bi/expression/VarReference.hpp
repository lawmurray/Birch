/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/expression/VarParameter.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Reference.hpp"

namespace bi {
/**
 * Reference to a variable.
 *
 * @ingroup compiler_expression
 */
class VarReference: public Expression, public Named, public Reference<
    VarParameter> {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param loc Location.
   * @param target Target.
   */
  VarReference(shared_ptr<Name> name, shared_ptr<Location> loc = nullptr,
      VarParameter* target = nullptr);

  /**
   * Destructor.
   */
  virtual ~VarReference();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool dispatchDefinitely(Expression& o);
  virtual bool definitely(VarReference& o);
  virtual bool definitely(VarParameter& o);

  virtual bool dispatchPossibly(Expression& o);
  virtual bool possibly(VarReference& o);
  virtual bool possibly(VarParameter& o);
};
}
