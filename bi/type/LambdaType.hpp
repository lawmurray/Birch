/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"

#include "bi/primitive/unique_ptr.hpp"

namespace bi {
/**
 * Lambda function type.
 *
 * @ingroup compiler_type
 */
class LambdaType: public Type {
public:
  /**
   * Constructor.
   *
   * @param parens Parameters type.
   * @param result Result type.
   * @param loc Location.
   * @param assignable Is this type writeable?
   */
  LambdaType(Type* parens, Type* result, shared_ptr<Location> loc = nullptr,
      const bool assignable = false);

  /**
   * Destructor.
   */
  virtual ~LambdaType();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool isLambda() const;

  virtual bool dispatchDefinitely(Type& o);
  virtual bool definitely(LambdaType& o);

  virtual bool dispatchPossibly(Type& o);
  virtual bool possibly(LambdaType& o);

  /**
   * Parameters type.
   */
  unique_ptr<Type> parens;

  /**
   * Result type.
   */
  unique_ptr<Type> result;
};
}
