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

  virtual bool isBuiltin() const;
  virtual bool isModel() const;

  virtual Type* strip();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Type::definitely;
  using Type::possibly;

  virtual bool definitely(Type& o);
  virtual bool dispatchDefinitely(Type& o);

  virtual bool possibly(Type& o);
  virtual bool dispatchPossibly(Type& o);
};
}
