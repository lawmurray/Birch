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
   * @param assignable Is this type assignable?
   * @param polymorphic Is this type polymorphic?
   */
  BracketsType(Type* single, Expression* brackets, shared_ptr<Location> loc =
      nullptr, const bool assignable = false,
      const bool polymorphic = false);

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
  virtual bool isArray() const;

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Type::definitely;
  using Type::possibly;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const BracketsType& o) const;

  virtual bool dispatchPossibly(const Type& o) const;
  virtual bool possibly(const BracketsType& o) const;

  /**
   * Number of dimensions.
   */
  int ndims;
};
}
