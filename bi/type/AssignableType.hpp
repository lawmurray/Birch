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
   * @param assignable Is this type writeable?
   */
  AssignableType(Type* single, shared_ptr<Location> loc = nullptr,
      const bool assignable = false);

  /**
   * Destructor.
   */
  virtual ~AssignableType();

  virtual bool isBuiltin() const;
  virtual bool isModel() const;
  virtual bool isDelay() const;
  virtual bool isLambda() const;

  virtual Type* strip();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const AssignableType& o) const;
  virtual bool definitely(const BracketsType& o) const;
  virtual bool definitely(const EmptyType& o) const;
  virtual bool definitely(const LambdaType& o) const;
  virtual bool definitely(const List<Type>& o) const;
  virtual bool definitely(const ModelParameter& o) const;
  virtual bool definitely(const ModelReference& o) const;
  virtual bool definitely(const ParenthesesType& o) const;
  virtual bool definitely(const DelayType& o) const;
  virtual bool definitely(const VariantType& o) const;

  virtual bool dispatchPossibly(const Type& o) const;
  virtual bool possibly(const AssignableType& o) const;
  virtual bool possibly(const BracketsType& o) const;
  virtual bool possibly(const EmptyType& o) const;
  virtual bool possibly(const LambdaType& o) const;
  virtual bool possibly(const List<Type>& o) const;
  virtual bool possibly(const ModelParameter& o) const;
  virtual bool possibly(const ModelReference& o) const;
  virtual bool possibly(const ParenthesesType& o) const;
  virtual bool possibly(const DelayType& o) const;
  virtual bool possibly(const VariantType& o) const;
};
}
