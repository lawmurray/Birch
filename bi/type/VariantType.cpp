/**
 * @file
 */
#include "bi/type/VariantType.hpp"

#include "bi/type/ModelParameter.hpp"
#include "bi/visitor/all.hpp"

#include <algorithm>

bi::VariantType::VariantType(Type* definite,
    const std::list<Type*>& possibles, shared_ptr<Location> loc) :
    Type(loc),
    definite(definite),
    possibles(possibles) {
  //
}

bi::VariantType::VariantType(Type* definite, shared_ptr<Location> loc) :
    Type(loc),
    definite(definite) {
  //
}

bi::VariantType::~VariantType() {
  //
}

void bi::VariantType::add(Type* o) {
  auto f = [&](Type* possible) {
    return possible->definitely(*o) || o->definitely(*possible);
  };
  bool exists = definite->definitely(*o) && o->definitely(*definite);
  exists = exists || std::find_if(possibles.begin(), possibles.end(), f)
          != possibles.end();
  if (!exists) {
    possibles.push_back(o);
  }
}

int bi::VariantType::size() const {
  return possibles.size() + 1;
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
  return definite->definitely(o);
}

bool bi::VariantType::dispatchDefinitely(Type& o) {
  return definite->dispatchDefinitely(o);
}

bool bi::VariantType::possibly(Type& o) {
  auto f = [&](Type* possible) {
    return possible->definitely(o) || possible->possibly(o);
  };
  return definite->possibly(o)
      || std::find_if(possibles.begin(), possibles.end(), f)
          != possibles.end();
}

bool bi::VariantType::dispatchPossibly(Type& o) {
  auto f = [&](Type* possible) {
    return possible->dispatchDefinitely(o) || possible->dispatchPossibly(o);
  };
  return definite->dispatchPossibly(o)
      || std::find_if(possibles.begin(), possibles.end(), f)
          != possibles.end();
}
