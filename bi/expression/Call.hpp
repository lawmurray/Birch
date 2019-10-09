/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Single.hpp"
#include "bi/common/Argumented.hpp"
#include "bi/common/Reference.hpp"
#include "bi/common/Unknown.hpp"

namespace bi {
/**
 * Call to a function.
 *
 * @ingroup expression
 *
 * @tparam ObjectType The particular type of object referred to by the
 * identifier.
 */
template<class ObjectType = Unknown>
class Call: public Expression,
    public Single<Expression>,
    public Argumented,
    public Reference<ObjectType> {
public:
  /**
   * Constructor.
   *
   * @param single Expression indicating the function.
   * @param args Arguments.
   * @param loc Location.
   */
  Call(Expression* single, Expression* args, Location* loc = nullptr);

  /**
   * Constructor for call with no arguments.
   *
   * @param single Expression indicating the function.
   * @param loc Location.
   */
  Call(Expression* single, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Call();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
