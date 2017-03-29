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
  virtual bool isRandom() const;
  virtual bool isLambda() const;

  virtual Type* strip();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool dispatchDefinitely(Type& o);
  virtual bool definitely(AssignableType& o);
  virtual bool definitely(BracketsType& o);
  virtual bool definitely(EmptyType& o);
  virtual bool definitely(LambdaType& o);
  virtual bool definitely(List<Type>& o);
  virtual bool definitely(ModelParameter& o);
  virtual bool definitely(ModelReference& o);
  virtual bool definitely(ParenthesesType& o);
  virtual bool definitely(RandomType& o);
  virtual bool definitely(VariantType& o);

  virtual bool dispatchPossibly(Type& o);
  virtual bool possibly(AssignableType& o);
  virtual bool possibly(BracketsType& o);
  virtual bool possibly(EmptyType& o);
  virtual bool possibly(LambdaType& o);
  virtual bool possibly(List<Type>& o);
  virtual bool possibly(ModelParameter& o);
  virtual bool possibly(ModelReference& o);
  virtual bool possibly(ParenthesesType& o);
  virtual bool possibly(RandomType& o);
  virtual bool possibly(VariantType& o);
};
}
