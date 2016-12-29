/**
 * @file
 */
#include "bi/type/EmptyType.hpp"

#include "bi/visitor/all.hpp"

bi::EmptyType::EmptyType(shared_ptr<Location> loc) :
    Type(loc) {
  //
}

bi::Type* bi::EmptyType::acceptClone(Cloner* visitor) const {
  return visitor->clone(this);
}

void bi::EmptyType::acceptModify(Modifier* visitor) {
  visitor->modify(this);
}

void bi::EmptyType::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::EmptyType::operator<=(Type& o) {
  try {
    EmptyType& o1 = dynamic_cast<EmptyType&>(o);
    return true;
  } catch (std::bad_cast e) {
    //
  }
  try {
    ParenthesesType& o1 = dynamic_cast<ParenthesesType&>(o);
    return *this <= *o1.type;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

bool bi::EmptyType::operator==(const Type& o) const {
  try {
    const EmptyType& o1 = dynamic_cast<const EmptyType&>(o);
    return true;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}
