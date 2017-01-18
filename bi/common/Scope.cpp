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
  return funcs.contains(param);
}

bool bi::Scope::contains(ModelParameter* param) {
  return models.contains(param);
}

bool bi::Scope::contains(ProgParameter* param) {
  return progs.contains(param);
}

void bi::Scope::add(VarParameter* param) {
  vars.add(param);
}

void bi::Scope::add(FuncParameter* param) {
  funcs.add(param);
}

void bi::Scope::add(ModelParameter* param) {
  models.add(param);
}

void bi::Scope::add(ProgParameter* prog) {
  progs.add(prog);
}

void bi::Scope::resolve(VarReference* ref) {
  vars.resolve(ref);
  if (!ref->target) {
    resolveDefer<VarParameter,VarReference>(ref);
  }
}

void bi::Scope::resolve(FuncReference* ref) {
  funcs.resolve(ref);
  if (!ref->target) {
    resolveDefer<FuncParameter,FuncReference>(ref);
  }
}

void bi::Scope::resolve(ModelReference* ref) {
  models.resolve(ref);
  if (!ref->target) {
    resolveDefer<ModelParameter,ModelReference>(ref);
  }
}

void bi::Scope::import(Scope* scope) {
  imports.insert(scope);
}
