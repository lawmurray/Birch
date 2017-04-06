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
   * @param assignable Is this type assignable?
   * @param polymorphic Is this type polymorphic?
   */
  LambdaType(Type* parens, Type* result, shared_ptr<Location> loc = nullptr,
      const bool assignable = false,
      const bool polymorphic = false);

  /**
   * Destructor.
   */
  virtual ~LambdaType();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool isLambda() const;

  using Type::definitely;
  using Type::possibly;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const LambdaType& o) const;
  virtual bool definitely(const ParenthesesType& o) const;

  virtual bool dispatchPossibly(const Type& o) const;
  virtual bool possibly(const LambdaType& o) const;
  virtual bool possibly(const ParenthesesType& o) const;

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
