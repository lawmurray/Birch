/**
 * @file
 */
#include "bi/common/Scope.hpp"

#include "bi/expression/VarParameter.hpp"
#include "bi/expression/FuncParameter.hpp"
#include "bi/expression/BinaryParameter.hpp"
#include "bi/expression/UnaryParameter.hpp"
#include "bi/statement/AssignmentParameter.hpp"
#include "bi/expression/ConversionParameter.hpp"
#include "bi/type/TypeParameter.hpp"

#include "bi/expression/VarReference.hpp"
#include "bi/expression/FuncReference.hpp"
#include "bi/expression/BinaryReference.hpp"
#include "bi/expression/UnaryReference.hpp"
#include "bi/statement/AssignmentReference.hpp"
#include "bi/type/TypeReference.hpp"

#include "bi/exception/all.hpp"
#include "bi/visitor/Cloner.hpp"

bool bi::Scope::contains(VarParameter* param) {
  return vars.contains(param);
}

bool bi::Scope::contains(FuncParameter* param) {
  return funcs.contains(param);
}

bool bi::Scope::contains(BinaryParameter* param) {
  return binaries.contains(param);
}

bool bi::Scope::contains(UnaryParameter* param) {
  return unaries.contains(param);
}

bool bi::Scope::contains(AssignmentParameter* param) {
  return assigns.contains(param);
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

void bi::Scope::add(BinaryParameter* param) {
  if (binaries.contains(param)) {
    throw PreviousDeclarationException(param, binaries.get(param));
  } else {
    binaries.add(param);
  }
}

void bi::Scope::add(UnaryParameter* param) {
  if (unaries.contains(param)) {
    throw PreviousDeclarationException(param, unaries.get(param));
  } else {
    unaries.add(param);
  }
}

void bi::Scope::add(AssignmentParameter* param) {
  if (assigns.contains(param)) {
    throw PreviousDeclarationException(param, assigns.get(param));
  } else {
    assigns.add(param);
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
  vars.resolve(ref);
  if (ref->matches.size() == 0) {
    resolveDefer(ref);
  }
}

void bi::Scope::resolve(FuncReference* ref) {
  funcs.resolve(ref);
  if (ref->matches.size() == 0) {
    resolveDefer(ref);
  }
}

void bi::Scope::resolve(BinaryReference* ref) {
  binaries.resolve(ref);
  if (ref->matches.size() == 0) {
    resolveDefer(ref);
  }
}

void bi::Scope::resolve(UnaryReference* ref) {
  unaries.resolve(ref);
  if (ref->matches.size() == 0) {
    resolveDefer(ref);
  }
}

void bi::Scope::resolve(AssignmentReference* ref) {
  assigns.resolve(ref);
  if (ref->matches.size() == 0) {
    resolveDefer(ref);
  }
}

void bi::Scope::resolve(TypeReference* ref) {
  types.resolve(ref);
  if (ref->matches.size() == 0) {
    resolveDefer(ref);
  }
}

void bi::Scope::inherit(Scope* scope) {
  bases.insert(scope);
}

void bi::Scope::import(Scope* scope) {
  vars.merge(scope->vars);
  funcs.merge(scope->funcs);
  binaries.merge(scope->binaries);
  unaries.merge(scope->unaries);
  assigns.merge(scope->assigns);
  convs.merge(scope->convs);
  types.merge(scope->types);
  progs.merge(scope->progs);
}
