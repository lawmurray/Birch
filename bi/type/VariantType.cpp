/**
 * @file
 */
#include "bi/type/VariantType.hpp"

#include "bi/type/ModelParameter.hpp"
#include "bi/visitor/all.hpp"

#include <algorithm>

bi::VariantType::VariantType(const std::list<Type*>& types,
    shared_ptr<Location> loc) :
    Type(loc),
    types(types) {
  //
}

bi::VariantType::VariantType(shared_ptr<Location> loc) :
    Type(loc) {
  //
}

bi::VariantType::~VariantType() {
  //
}

void bi::VariantType::add(Type* o) {
  Type* o1 = o->strip();
  auto f = [&](Type* type) {
    return o1->equals(*type);
  };
  auto iter = std::find_if(types.begin(), types.end(), f);
  if (iter == types.end()) {
    types.push_back(o1);
  }
}

int bi::VariantType::size() const {
  return types.size();
}

bi::Type* bi::VariantType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::VariantType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::VariantType::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::VariantType::definitelyAll(Type& o) {
  auto f = [&](Type* type) {
    return type->definitely(o);
  };
  return std::all_of(types.begin(), types.end(), f);
}

bool bi::VariantType::possiblyAny(Type& o) {
  auto f = [&](Type* type) {
    return type->possibly(o);
  };
  return std::any_of(types.begin(), types.end(), f);
}

bool bi::VariantType::dispatchDefinitely(Type& o) {
  auto f = [&](Type* type) {
    return type->dispatchDefinitely(o);
  };
  return std::all_of(types.begin(), types.end(), f);
}

bool bi::VariantType::definitely(AssignableType& o) {
  return definitelyAll(o);
}

bool bi::VariantType::definitely(BracketsType& o) {
  return definitelyAll(o);
}

bool bi::VariantType::definitely(EmptyType& o) {
  return definitelyAll(o);
}

bool bi::VariantType::definitely(LambdaType& o) {
  return definitelyAll(o);
}

bool bi::VariantType::definitely(List<Type>& o) {
  return definitelyAll(o);
}

bool bi::VariantType::definitely(ModelParameter& o) {
  return definitelyAll(o);
}

bool bi::VariantType::definitely(ModelReference& o) {
  return definitelyAll(o);
}

bool bi::VariantType::definitely(ParenthesesType& o) {
  return definitelyAll(o);
}

bool bi::VariantType::definitely(RandomType& o) {
  return definitelyAll(o);
}

bool bi::VariantType::definitely(VariantType& o) {
  return definitelyAll(o);
}

bool bi::VariantType::dispatchPossibly(Type& o) {
  auto f = [&](Type* type) {
    return type->dispatchPossibly(o);
  };
  return std::any_of(types.begin(), types.end(), f);
}

bool bi::VariantType::possibly(AssignableType& o) {
  return possiblyAny(o);
}

bool bi::VariantType::possibly(BracketsType& o) {
  return possiblyAny(o);
}

bool bi::VariantType::possibly(EmptyType& o) {
  return possiblyAny(o);
}

bool bi::VariantType::possibly(LambdaType& o) {
  return possiblyAny(o);
}

bool bi::VariantType::possibly(List<Type>& o) {
  return possiblyAny(o);
}

bool bi::VariantType::possibly(ModelParameter& o) {
  return possiblyAny(o);
}

bool bi::VariantType::possibly(ModelReference& o) {
  return possiblyAny(o);
}

bool bi::VariantType::possibly(ParenthesesType& o) {
  return possiblyAny(o);
}

bool bi::VariantType::possibly(RandomType& o) {
  return possiblyAny(o);
}

bool bi::VariantType::possibly(VariantType& o) {
  return possiblyAny(o);
}
