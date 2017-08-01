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
 * @ingroup compiler_type
 */
class ParenthesesType: public Type, public Single<Type> {
public:
  /**
   * Constructor.
   *
   * @param single Type in parentheses.
   * @param loc Location.
   * @param assignable Is this type assignable?
   */
  ParenthesesType(Type* single, Location* loc = nullptr,
      const bool assignable = false);

  /**
   * Destructor.
   */
  virtual ~ParenthesesType();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const ParenthesesType& o) const;

  virtual bool dispatchPossibly(const Type& o) const;
  virtual bool possibly(const ParenthesesType& o) const;
};
}
