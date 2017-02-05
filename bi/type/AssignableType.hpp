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

  bool isBuiltin() const;
  bool isModel() const;

  virtual Type* strip();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool dispatchDefinitely(Type& o);
  virtual bool definitely(AssignableType& o);
  virtual bool definitely(BracketsType& o);
  virtual bool definitely(EmptyType& o);
  virtual bool definitely(List<Type>& o);
  virtual bool definitely(ModelParameter& o);
  virtual bool definitely(ModelReference& o);
  virtual bool definitely(ParenthesesType& o);

  virtual bool dispatchPossibly(Type& o);
  virtual bool possibly(AssignableType& o);
  virtual bool possibly(BracketsType& o);
  virtual bool possibly(EmptyType& o);
  virtual bool possibly(List<Type>& o);
  virtual bool possibly(ModelParameter& o);
  virtual bool possibly(ModelReference& o);
  virtual bool possibly(ParenthesesType& o);
};
}
