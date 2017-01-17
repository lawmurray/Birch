/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/Unary.hpp"
#include "bi/primitive/unique_ptr.hpp"

namespace bi {
/**
 * Tuple type.
 *
 * @ingroup compiler_type
 */
class ParenthesesType: public Type, public TypeUnary {
public:
  /**
   * Constructor.
   *
   * @param single Type in parentheses.
   * @param loc Location.
   */
  ParenthesesType(Type* single, shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~ParenthesesType();

  virtual Type* strip();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual possibly dispatch(Type& o);
  virtual possibly le(EmptyType& o);
  virtual possibly le(List<Type>& o);
  virtual possibly le(ModelParameter& o);
  virtual possibly le(ModelReference& o);
  virtual possibly le(ParenthesesType& o);
  virtual possibly le(RandomType& o);
};
}
