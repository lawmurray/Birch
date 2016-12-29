/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/primitive/unique_ptr.hpp"

namespace bi {
/**
 * ParenthesesType.
 *
 * @ingroup compiler_type
 */
class ParenthesesType: public Type {
public:
  /**
   * Constructor.
   *
   * @param type Type in parentheses.
   * @param loc Location.
   */
  ParenthesesType(Type* type, shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~ParenthesesType();

  virtual Type* acceptClone(Cloner* visitor) const;
  virtual void acceptModify(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool operator<=(Type& o);
  virtual bool operator==(const Type& o) const;

  /**
   * Expression inside parentheses.
   */
  unique_ptr<Type> type;
};
}

inline bi::ParenthesesType::~ParenthesesType() {
  //
}
