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

bi::VarParameter* bi::Scope::resolve(VarReference* ref) {
  VarParameter* result = vars.resolve(ref);
  if (!result) {
    result = resolveDefer<VarParameter,VarReference>(ref);
  }
  return result;
}

bi::FuncParameter* bi::Scope::resolve(FuncReference* ref) {
  FuncParameter* result = funcs.resolve(ref);
  if (!result) {
    result = resolveDefer<FuncParameter,FuncReference>(ref);
  }
  return result;
}

bi::ModelParameter* bi::Scope::resolve(ModelReference* ref) {
  ModelParameter* result = models.resolve(ref);
  if (!result) {
    result = resolveDefer<ModelParameter,ModelReference>(ref);
  }
  return result;
}

void bi::Scope::import(Scope* scope) {
  imports.insert(scope);
}
