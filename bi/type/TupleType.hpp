/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/Single.hpp"

namespace bi {
/**
 * Tuple type.
 *
 * @ingroup type
 */
class TupleType: public Type, public Single<Type> {
public:
  /**
   * Constructor.
   *
   * @param single Type in parentheses.
   * @param loc Location.
   */
  TupleType(Type* single, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~TupleType();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool isTuple() const;
};
}
