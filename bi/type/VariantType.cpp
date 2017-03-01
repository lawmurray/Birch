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
  auto f = [&](Type* type) {
    return *type == *o;
  };
  auto iter = std::find_if(types.begin(), types.end(), f);
  if (iter == types.end()) {
    types.push_back(o);
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

bool bi::VariantType::definitely(Type& o) {
  auto f = [&](Type* type) {
    return type->definitely(o);
  };
  return std::find_if(types.begin(), types.end(), f) != types.end();
}

bool bi::VariantType::dispatchDefinitely(Type& o) {
  auto f = [&](Type* type) {
    return type->dispatchDefinitely(o);
  };
  return std::find_if(types.begin(), types.end(), f) != types.end();
}

bool bi::VariantType::possibly(Type& o) {
  auto f = [&](Type* type) {
    return type->possibly(o);
  };
  return std::find_if(types.begin(), types.end(), f) != types.end();
}

bool bi::VariantType::dispatchPossibly(Type& o) {
  auto f = [&](Type* type) {
    return type->dispatchPossibly(o);
  };
  return std::find_if(types.begin(), types.end(), f) != types.end();
}
