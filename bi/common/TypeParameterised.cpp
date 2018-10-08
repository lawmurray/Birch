/**
 * @file
 */
#include "bi/common/TypeParameterised.hpp"

#include "bi/visitor/all.hpp"

template<class Target>
bi::TypeParameterised<Target>::TypeParameterised(Expression* typeParams) :
    typeParams(typeParams),
    isExplicit(false),
    state(CLONED) {
  //
}

template<class Target>
bi::TypeParameterised<Target>::~TypeParameterised() {
  //
}

template<class Target>
bool bi::TypeParameterised<Target>::isGeneric() const {
  return !typeParams->isEmpty();
}

template<class Target>
bool bi::TypeParameterised<Target>::isBound() const {
  for (auto param : *typeParams) {
    if (param->type->isEmpty()) {
      return false;
    }
  }
  return true;
}

template<class Target>
bool bi::TypeParameterised<Target>::isInstantiation() const {
  return isGeneric() && isBound();
}

template<class Target>
void bi::TypeParameterised<Target>::bind(Type* typeArgs) {
  assert(typeArgs->width() == typeParams->width());

  Cloner cloner;
  auto arg = typeArgs->begin();
  auto param = typeParams->begin();
  while (arg != typeArgs->end() && param != typeParams->end()) {
    (*param)->type = (*arg)->canonical()->accept(&cloner);
    ++arg;
    ++param;
  }
}

template<class Target>
void bi::TypeParameterised<Target>::addInstantiation(Target* o) {
  instantiations.push_back(o);
}

template<class Target>
Target* bi::TypeParameterised<Target>::getInstantiation(const Type* typeArgs) {
  auto compare = [](const Type* arg, const Expression* param) {
    return arg->equals(*param->type);
  };
  for (auto o : instantiations) {
    bool matches = typeArgs->width() == o->typeParams->width()
    && std::equal(typeArgs->begin(), typeArgs->end(),
        o->typeParams->begin(), compare);
    if (matches) {
      return o;
    }
  }
  return nullptr;
}

template class bi::TypeParameterised<bi::Class>;
template class bi::TypeParameterised<bi::Function>;
template class bi::TypeParameterised<bi::Fiber>;
template class bi::TypeParameterised<bi::MemberFunction>;
template class bi::TypeParameterised<bi::MemberFiber>;
template class bi::TypeParameterised<bi::BinaryOperator>;
template class bi::TypeParameterised<bi::UnaryOperator>;
