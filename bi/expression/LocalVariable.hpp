/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/expression/EmptyExpression.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Numbered.hpp"
#include "bi/primitive/unique_ptr.hpp"

namespace bi {
/**
 * Global variable.
 *
 * @ingroup compiler_expression
 */
class LocalVariable: public Expression,
    public Named,
    public Numbered {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param type Type.
   * @param loc Location.
   */
  LocalVariable(shared_ptr<Name> name, Type* type, shared_ptr<Location> loc =
      nullptr);

  /**
   * Destructor.
   */
  virtual ~LocalVariable();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Expression::definitely;
  using Expression::possibly;

  virtual bool dispatchDefinitely(const Expression& o) const;
  virtual bool definitely(const LocalVariable& o) const;

  virtual bool dispatchPossibly(const Expression& o) const;
  virtual bool possibly(const LocalVariable& o) const;
};
}
