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
   * @param result Result type.
   * @param loc Location.
   */
  LambdaType(Type* result, shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~LambdaType();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool isLambda() const;

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

  /**
   * Result type.
   */
  unique_ptr<Type> result;
};
}
