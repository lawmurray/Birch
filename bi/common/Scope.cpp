/**
 * @file
 */
#include "bi/common/Scope.hpp"

#include "bi/expression/VarParameter.hpp"
#include "bi/expression/FuncParameter.hpp"
#include "bi/type/ModelParameter.hpp"
#include "bi/exception/all.hpp"
#include "bi/visitor/Cloner.hpp"

bool bi::Scope::contains(VarParameter* param) {
  return vars.contains(param);
}

bool bi::Scope::contains(FuncParameter* param) {
  return definites.contains(param);
}

bool bi::Scope::contains(ModelParameter* param) {
  return models.contains(param);
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
  if (definites.contains(param)) {
    throw PreviousDeclarationException(param, definites.get(param));
  } else {
    definites.add(param);
    possibles.add(param);
  }
}

void bi::Scope::add(ModelParameter* param) {
  if (models.contains(param)) {
    throw PreviousDeclarationException(param, models.get(param));
  } else {
    models.add(param);
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
  ref->target = definites.resolve(ref);
  if (!ref->target) {
    resolveDefer<FuncParameter,FuncReference>(ref);
  } else {
    possibles.resolve(ref, ref->possibles);
  }
}

void bi::Scope::resolve(ModelReference* ref) {
  ref->target = models.resolve(ref);
  if (!ref->target) {
    resolveDefer<ModelParameter,ModelReference>(ref);
  }
}

bool bi::Scope::contains(Dispatcher* dispatcher) {
  return dispatchers.contains(dispatcher);
}

void bi::Scope::add(Dispatcher* dispatcher) {
  dispatchers.add(dispatcher);
}

bi::Dispatcher* bi::Scope::get(Dispatcher* dispatcher) {
  return dispatchers.get(dispatcher);
}

void bi::Scope::inherit(Scope* scope) {
  bases.insert(scope);
}

void bi::Scope::import(Scope* scope) {
  vars.merge(scope->vars);
  definites.merge(scope->definites);
  possibles.merge(scope->possibles);
  models.merge(scope->models);
  progs.merge(scope->progs);
}
