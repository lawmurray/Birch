/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Named.hpp"
#include "bi/expression/VarParameter.hpp"

namespace bi {
/**
 * Self-reference to an object.
 *
 * @ingroup compiler_expression
 */
class RandomRight: public Expression, public Named {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param loc Location.
   */
  RandomRight(shared_ptr<Name> name, shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~RandomRight();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool dispatch(Expression& o);
  virtual bool le(RandomRight& o);
  virtual bool le(VarParameter& o);
};
}
