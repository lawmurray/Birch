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

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool dispatch(Type& o);
  virtual bool le(EmptyType& o);
  virtual bool le(List<Type>& o);
  virtual bool le(ModelParameter& o);
  virtual bool le(ModelReference& o);
  virtual bool le(ParenthesesType& o);
  virtual bool le(RandomType& o);
};
}
