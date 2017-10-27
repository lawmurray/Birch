/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/ReturnTyped.hpp"

namespace bi {
/**
 * Function type.
 *
 * @ingroup compiler_type
 */
class FunctionType: public Type, public ReturnTyped {
public:
  /**
   * Constructor.
   *
   * @param params Parameters type.
   * @param returnType Return type.
   * @param loc Location.
   * @param assignable Is this type assignable?
   */
  FunctionType(Type* params, Type* returnType, Location* loc = nullptr,
      const bool assignable = false);

  /**
   * Destructor.
   */
  virtual ~FunctionType();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool isFunction() const;

  virtual FunctionType* resolve(Argumented* o);

  using Type::definitely;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const AliasType& o) const;
  virtual bool definitely(const GenericType& o) const;
  virtual bool definitely(const FunctionType& o) const;
  virtual bool definitely(const OptionalType& o) const;

  /**
   * Parameters type.
   */
  Type* params;
};
}
