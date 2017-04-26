/**
 * @file
 */
#include "bi/common/Scope.hpp"

#include "bi/expression/VarParameter.hpp"
#include "bi/expression/FuncParameter.hpp"
#include "bi/type/TypeParameter.hpp"
#include "bi/exception/all.hpp"
#include "bi/visitor/Cloner.hpp"

#include <vector>

bool bi::Scope::contains(VarParameter* param) {
  return vars.contains(param);
}

bool bi::Scope::contains(FuncParameter* param) {
  return definites.contains(param);
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
  if (definites.contains(param)) {
    throw PreviousDeclarationException(param, definites.get(param));
  } else {
    definites.add(param);
    possibles.add(param);
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
  ref->target = definites.resolve(ref);
  if (!ref->target) {
    resolveDefer<FuncParameter,FuncReference>(ref);
  } else {
    /* find all definite and possible matches */
    std::list<FuncParameter*> definites1, possibles1;
    definites.resolve(ref, definites1);
    possibles.resolve(ref, possibles1);

    /* remove any definite matches in the list of possible matches, while
     * preserving the order of possible matches (specifically, don't use
     * std::set_difference(), as it requires sorting of the two lists first,
     * which destroys the required order) */
    for (auto iter = definites1.begin(); iter != definites1.end(); ++iter) {
      auto find = std::find(possibles1.begin(), possibles1.end(), *iter);
      if (find != possibles1.end()) {
        possibles1.erase(find);
      }
    }
    ref->possibles = possibles1;
  }
}

void bi::Scope::resolve(TypeReference* ref) {
  ref->target = types.resolve(ref);
  if (!ref->target) {
    resolveDefer<TypeParameter,TypeReference>(ref);
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
  types.merge(scope->types);
  progs.merge(scope->progs);
}
