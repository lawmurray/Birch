/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/Unary.hpp"

namespace bi {
/**
 * Modifier to force a type to be assignable.
 *
 * @ingroup compiler_type
 */
class AssignableType: public Type, public TypeUnary {
public:
  /**
   * Constructor.
   *
   * @param single Type.
   * @param loc Location.
   */
  AssignableType(Type* single, shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~AssignableType();

  virtual bool isBuiltin() const;
  virtual bool isModel() const;
  virtual bool isRandom() const;
  virtual bool isLambda() const;

  virtual Type* strip();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Type::definitely;
  using Type::possibly;

  virtual bool definitely(Type& o);
  virtual bool dispatchDefinitely(Type& o);

  virtual bool possibly(Type& o);
  virtual bool dispatchPossibly(Type& o);
};
}
