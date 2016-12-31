/**
 * @file
 */
#include "bi/type/ParenthesesType.hpp"

#include "bi/type/ModelParameter.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::ParenthesesType::ParenthesesType(Type* type, shared_ptr<Location> loc) :
    Type(loc), type(type) {
  /* pre-condition */
  assert(type);
}

bi::ParenthesesType::~ParenthesesType() {
  //
}

bi::Type* bi::ParenthesesType::acceptClone(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::ParenthesesType::acceptModify(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::ParenthesesType::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::ParenthesesType::operator<=(Type& o) {
  try {
    ParenthesesType& o1 = dynamic_cast<ParenthesesType&>(o);
    return *type <= *o1.type;
  } catch (std::bad_cast e) {
    //
  }

  /* parentheses may be used unnecessarily in situations where precedence is
   * clear anyway; accommodate these by making their use optional in
   * matches */
  return *type <= o;
}

bool bi::ParenthesesType::operator==(const Type& o) const {
  return *type == o;
}
