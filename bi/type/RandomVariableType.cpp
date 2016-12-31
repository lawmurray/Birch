/**
 * @file
 */
#include "bi/type/RandomVariableType.hpp"

#include "bi/type/ModelParameter.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::RandomVariableType::RandomVariableType(Type* left, Type* right,
    shared_ptr<Location> loc) :
    Type(loc),
    TypeBinary(left, right) {
  //
}

bi::RandomVariableType::~RandomVariableType() {
  //
}

bi::Type* bi::RandomVariableType::acceptClone(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::RandomVariableType::acceptModify(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::RandomVariableType::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::RandomVariableType::operator<=(Type& o) {
  try {
    RandomVariableType& o1 = dynamic_cast<RandomVariableType&>(o);
    return *left <= *o1.left && *right <= *o1.right;
  } catch (std::bad_cast e) {
    //
  }
  try {
    ModelParameter& o1 = dynamic_cast<ModelParameter&>(o);
    return *this <= *o1.base && o1.capture(this);
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

bool bi::RandomVariableType::operator==(const Type& o) const {
  try {
    const RandomVariableType& o1 = dynamic_cast<const RandomVariableType&>(o);
    return *left == *o1.left && *right == *o1.right;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}
