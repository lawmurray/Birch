/**
 * @file
 */
#include "bi/type/RandomType.hpp"

#include "bi/type/ModelParameter.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::RandomType::RandomType(Type* left, Type* right,
    shared_ptr<Location> loc) :
    Type(loc),
    TypeBinary(left, right) {
  //
}

bi::RandomType::~RandomType() {
  //
}

bi::Type* bi::RandomType::acceptClone(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::RandomType::acceptModify(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::RandomType::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::RandomType::operator<=(Type& o) {
  try {
    RandomType& o1 = dynamic_cast<RandomType&>(o);
    return *left <= *o1.left && *right <= *o1.right;
  } catch (std::bad_cast e) {
    //
  }
  return *left <= o; // can also treat as variate alone
}

bool bi::RandomType::operator==(const Type& o) const {
  try {
    const RandomType& o1 = dynamic_cast<const RandomType&>(o);
    return *left == *o1.left && *right == *o1.right;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}
