/**
 * @file
 */
#include "bi/type/RandomType.hpp"

#include "bi/type/ModelParameter.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::RandomType::RandomType(Type* left, Type* right, shared_ptr<Location> loc) :
    Type(loc),
    TypeBinary(left, right) {
  //
}

bi::RandomType::~RandomType() {
  //
}

bi::Type* bi::RandomType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::RandomType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::RandomType::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::RandomType::isRandom() const {
  return true;
}

bool bi::RandomType::dispatchDefinitely(Type& o) {
  return o.definitely(*this);
}

bool bi::RandomType::definitely(EmptyType& o) {
  return !o.assignable || assignable;
}

bool bi::RandomType::definitely(LambdaType& o) {
  return left->definitely(o) && (!o.assignable || assignable);
}

bool bi::RandomType::definitely(List<Type>& o) {
  return left->definitely(o) && (!o.assignable || assignable);
}

bool bi::RandomType::definitely(ModelReference& o) {
  return left->definitely(o) && (!o.assignable || assignable);
}

bool bi::RandomType::definitely(ModelParameter& o) {
  return left->definitely(o) && (!o.assignable || assignable);
}

bool bi::RandomType::dispatchPossibly(Type& o) {
  return o.possibly(*this);
}

bool bi::RandomType::possibly(EmptyType& o) {
  return !o.assignable || assignable;
}

bool bi::RandomType::possibly(LambdaType& o) {
  return (left->possibly(o) || right->possibly(o))
      && (!o.assignable || assignable);
}

bool bi::RandomType::possibly(List<Type>& o) {
  return (left->possibly(o) || right->possibly(o))
      && (!o.assignable || assignable);
}

bool bi::RandomType::possibly(ModelReference& o) {
  return (left->possibly(o) || right->possibly(o))
      && (!o.assignable || assignable);
}

bool bi::RandomType::possibly(ModelParameter& o) {
  return (left->possibly(o) || right->possibly(o))
      && (!o.assignable || assignable);
}

bool bi::RandomType::possibly(RandomType& o) {
  return left->possibly(*o.left) && right->possibly(*o.right)
      && (!o.assignable || assignable);
}

bool bi::RandomType::equals(Type& o) {
  /* RandomType objects are never definitely equal; this override is
   * something of a hack to ensure that, in the context of adding
   * RandomType objects to VariantType objects, each type appears only
   * once */
  try {
    RandomType& random = dynamic_cast<RandomType&>(*o.strip());
    return left->equals(*random.left) && right->equals(*random.right)
        && assignable == random.assignable;
  } catch (std::bad_cast) {
    return Type::equals(o);
  }
}
