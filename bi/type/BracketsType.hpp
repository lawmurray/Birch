/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/Bracketed.hpp"
#include "bi/common/Unary.hpp"
#include "bi/primitive/unique_ptr.hpp"

namespace bi {
/**
 * Bracketed type.
 *
 * @ingroup compiler_type
 */
class BracketsType: public Type, public TypeUnary, public Bracketed {
public:
  /**
   * Constructor.
   *
   * @param single Type.
   * @param brackets Brackets.
   * @param loc Location.
   */
  BracketsType(Type* single, Expression* brackets, shared_ptr<Location> loc =
      nullptr);

  /**
   * Constructor.
   *
   * @param single Type.
   * @param ndims Number of dimensions.
   */
  BracketsType(Type* single, const int ndims);

  /**
   * Destructor.
   */
  virtual ~BracketsType();

  virtual int count() const;

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual possibly dispatch(Type& o);
  virtual possibly le(BracketsType& o);
  virtual possibly le(AssignableType& o);
  virtual possibly le(ParenthesesType& o);

  /**
   * Number of dimensions.
   */
  int ndims;
};
}
