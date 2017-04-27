/**
 * @file
 */
#include "bi/common/Scope.hpp"

#include "bi/expression/VarParameter.hpp"
#include "bi/expression/FuncParameter.hpp"
#include "bi/expression/ConversionParameter.hpp"
#include "bi/type/TypeParameter.hpp"
#include "bi/exception/all.hpp"
#include "bi/visitor/Cloner.hpp"

#include <vector>

bool bi::Scope::contains(VarParameter* param) {
  return vars.contains(param);
}

bool bi::Scope::contains(FuncParameter* param) {
  return funcs.contains(param);
}

bool bi::Scope::contains(ConversionParameter* param) {
  return convs.contains(param);
}

bool bi::Scope::contains(TypeParameter* param) {
  return types.contains(param);
}

bool bi::Scope::contains(ProgParameter* param) {
  return progs.contains(param);
}

void bi::Scope::add(VarParameter* param) {
  if (vars.contains(param)) {
    throw PreviousDeclarationException(param, vars.get(param));
  } else {
    vars.add(param);
  }
}

void bi::Scope::add(FuncParameter* param) {
  if (funcs.contains(param)) {
    throw PreviousDeclarationException(param, funcs.get(param));
  } else {
    funcs.add(param);
  }
}

void bi::Scope::add(ConversionParameter* param) {
  if (convs.contains(param)) {
    throw PreviousDeclarationException(param, convs.get(param));
  } else {
    convs.add(param);
  }
}

void bi::Scope::add(TypeParameter* param) {
  if (types.contains(param)) {
    throw PreviousDeclarationException(param, types.get(param));
  } else {
    types.add(param);
  }
}

void bi::Scope::add(ProgParameter* param) {
  if (progs.contains(param)) {
    throw PreviousDeclarationException(param, progs.get(param));
  } else {
    progs.add(param);
  }
}

void bi::Scope::resolve(VarReference* ref) {
  ref->target = vars.resolve(ref);
  if (!ref->target) {
    resolveDefer<VarParameter,VarReference>(ref);
  }
}

void bi::Scope::resolve(FuncReference* ref) {
  ref->target = funcs.resolve(ref);
  if (!ref->target) {
    resolveDefer<FuncParameter,FuncReference>(ref);
  }
}

void bi::Scope::resolve(TypeReference* ref) {
  ref->target = types.resolve(ref);
  if (!ref->target) {
    resolveDefer<TypeParameter,TypeReference>(ref);
  }
}

void bi::Scope::inherit(Scope* scope) {
  bases.insert(scope);
}

void bi::Scope::import(Scope* scope) {
  vars.merge(scope->vars);
  funcs.merge(scope->funcs);
  convs.merge(scope->convs);
  types.merge(scope->types);
  progs.merge(scope->progs);
}
