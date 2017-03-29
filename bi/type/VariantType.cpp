/**
 * @file
 */
#include "bi/type/VariantType.hpp"

#include "bi/type/ModelParameter.hpp"
#include "bi/visitor/all.hpp"

#include <algorithm>

bi::VariantType::VariantType(Type* definite,
    const std::list<Type*>& possibles, shared_ptr<Location> loc,
    const bool assignable) :
    Type(loc, assignable),
    definite(definite),
    possibles(possibles) {
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
  if (!o1->equals(*definite)
      && !std::any_of(possibles.begin(), possibles.end(), f)) {
    possibles.push_back(o1);
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

bool bi::VariantType::isVariant() const {
  return true;
}

bool bi::VariantType::definitelyAll(Type& o) {
  return definite->definitely(o);
}

bool bi::VariantType::possiblyAny(Type& o) {
  auto f = [&](Type* type) {
    return type->possibly(o);
  };
  return definite->possibly(o)
      || std::any_of(possibles.begin(), possibles.end(), f);
}

bool bi::VariantType::dispatchDefinitely(Type& o) {
  auto f = [&](Type* type) {
    return type->dispatchDefinitely(o);
  };
  return definite->dispatchDefinitely(o)
      && std::all_of(possibles.begin(), possibles.end(), f);
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
  return definite->dispatchPossibly(o)
      || std::any_of(possibles.begin(), possibles.end(), f);
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
