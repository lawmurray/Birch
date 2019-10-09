/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Single.hpp"

namespace bi {
/**
 * Wrapper around an expression to be explicitly interpreted in the global
 * scope.
 *
 * @ingroup expression
 */
class Global: public Expression, public Single<Expression> {
public:
  /**
   * Constructor.
   *
   * @param single Expression.
   * @param loc Location.
   */
  Global(Expression* single, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Global();

  virtual Lookup lookup(Expression* args);
  virtual GlobalVariable* resolve(Call<GlobalVariable>* o);
  virtual Function* resolve(Call<Function>* o);
  virtual Fiber* resolve(Call<Fiber>* o);

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
