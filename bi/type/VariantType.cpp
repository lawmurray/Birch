/**
 * @file
 */
#include "bi/type/VariantType.hpp"

#include "bi/type/ModelParameter.hpp"
#include "bi/visitor/all.hpp"

#include <algorithm>

bi::VariantType::VariantType(Type* definite,
    const std::list<Type*>& possibles, shared_ptr<Location> loc,
    const bool assignable, const bool polymorphic) :
    Type(loc, assignable, polymorphic),
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
    return o1->definitely(*type);
  };
  if (!o1->definitely(*definite)
      && !std::any_of(possibles.begin(), possibles.end(), f)) {
    possibles.push_back(o1);
  } else {
    delete o;
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

bool bi::VariantType::definitelyAll(const Type& o) const {
  return definite->definitely(o);
}

bool bi::VariantType::possiblyAny(const Type& o) const {
  auto f = [&](Type* type) {
    return type->possibly(o);
  };
  return definite->possibly(o)
      || std::any_of(possibles.begin(), possibles.end(), f);
}

bool bi::VariantType::dispatchDefinitely(const Type& o) const {
  auto f = [&](Type* type) {
    return type->dispatchDefinitely(o);
  };
  return definite->dispatchDefinitely(o)
      && std::all_of(possibles.begin(), possibles.end(), f);
}

bool bi::VariantType::definitely(const BracketsType& o) const {
  return definitelyAll(o);
}

bool bi::VariantType::definitely(const EmptyType& o) const {
  return definitelyAll(o);
}

bool bi::VariantType::definitely(const LambdaType& o) const {
  return definitelyAll(o);
}

bool bi::VariantType::definitely(const List<Type>& o) const {
  return definitelyAll(o);
}

bool bi::VariantType::definitely(const ModelParameter& o) const {
  return definitelyAll(o);
}

bool bi::VariantType::definitely(const ModelReference& o) const {
  return definitelyAll(o);
}

bool bi::VariantType::definitely(const ParenthesesType& o) const {
  return definitelyAll(o);
}

bool bi::VariantType::definitely(const DelayType& o) const {
  return definitelyAll(o);
}

bool bi::VariantType::definitely(const VariantType& o) const {
  return definitelyAll(o);
}

bool bi::VariantType::dispatchPossibly(const Type& o) const {
  auto f = [&](Type* type) {
    return type->dispatchPossibly(o);
  };
  return definite->dispatchPossibly(o)
      || std::any_of(possibles.begin(), possibles.end(), f);
}

bool bi::VariantType::possibly(const BracketsType& o) const {
  return possiblyAny(o);
}

bool bi::VariantType::possibly(const EmptyType& o) const {
  return possiblyAny(o);
}

bool bi::VariantType::possibly(const LambdaType& o) const {
  return possiblyAny(o);
}

bool bi::VariantType::possibly(const List<Type>& o) const {
  return possiblyAny(o);
}

bool bi::VariantType::possibly(const ModelParameter& o) const {
  return possiblyAny(o);
}

bool bi::VariantType::possibly(const ModelReference& o) const {
  return possiblyAny(o);
}

bool bi::VariantType::possibly(const ParenthesesType& o) const {
  return possiblyAny(o);
}

bool bi::VariantType::possibly(const DelayType& o) const {
  return possiblyAny(o);
}

bool bi::VariantType::possibly(const VariantType& o) const {
  return possiblyAny(o);
}
