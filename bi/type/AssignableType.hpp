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

  virtual possibly dispatch(Type& o);
  virtual possibly le(AssignableType& o);
  virtual possibly le(BracketsType& o);
  virtual possibly le(EmptyType& o);
  virtual possibly le(List<Type>& o);
  virtual possibly le(ModelParameter& o);
  virtual possibly le(ModelReference& o);
  virtual possibly le(ParenthesesType& o);
  virtual possibly le(RandomType& o);
};
}
