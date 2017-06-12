/**
 * @file
 */
#include "bi/common/Scope.hpp"

#include "bi/expression/VarParameter.hpp"
#include "bi/statement/Function.hpp"
#include "bi/statement/Coroutine.hpp"
#include "bi/statement/Program.hpp"
#include "bi/statement/MemberFunction.hpp"
#include "bi/statement/BinaryOperator.hpp"
#include "bi/statement/UnaryOperator.hpp"
#include "bi/statement/AssignmentOperator.hpp"
#include "bi/statement/ConversionOperator.hpp"
#include "bi/type/TypeParameter.hpp"

#include "bi/expression/VarReference.hpp"
#include "bi/expression/FuncReference.hpp"
#include "bi/expression/BinaryReference.hpp"
#include "bi/expression/UnaryReference.hpp"
#include "bi/statement/Assignment.hpp"
#include "bi/type/TypeReference.hpp"

#include "bi/exception/all.hpp"
#include "bi/visitor/Cloner.hpp"

bool bi::Scope::contains(VarParameter* param) {
  return vars.contains(param);
}

bool bi::Scope::contains(Function* param) {
  return functions.contains(param);
}

bool bi::Scope::contains(Coroutine* param) {
  return coroutines.contains(param);
}

bool bi::Scope::contains(Program* param) {
  return programs.contains(param);
}

bool bi::Scope::contains(MemberFunction* param) {
  return memberFunctions.contains(param);
}

bool bi::Scope::contains(BinaryOperator* param) {
  return binaryOperators.contains(param);
}

bool bi::Scope::contains(UnaryOperator* param) {
  return unaryOperators.contains(param);
}

bool bi::Scope::contains(AssignmentOperator* param) {
  return assignmentOperators.contains(param);
}

bool bi::Scope::contains(ConversionOperator* param) {
  return conversionOperators.contains(param);
}

bool bi::Scope::contains(TypeParameter* param) {
  return types.contains(param);
}

void bi::Scope::add(VarParameter* param) {
  if (vars.contains(param)) {
    throw PreviousDeclarationException(param, vars.get(param));
  } else {
    vars.add(param);
  }
}

void bi::Scope::add(Function* param) {
  if (functions.contains(param)) {
    throw PreviousDeclarationException(param, functions.get(param));
  } else {
    functions.add(param);
  }
}

void bi::Scope::add(Coroutine* param) {
  if (coroutines.contains(param)) {
    throw PreviousDeclarationException(param, coroutines.get(param));
  } else {
    coroutines.add(param);
  }
}

void bi::Scope::add(Program* param) {
  if (programs.contains(param)) {
    throw PreviousDeclarationException(param, programs.get(param));
  } else {
    programs.add(param);
  }
}

void bi::Scope::add(MemberFunction* param) {
  if (memberFunctions.contains(param)) {
    throw PreviousDeclarationException(param, memberFunctions.get(param));
  } else {
    memberFunctions.add(param);
  }
}

void bi::Scope::add(BinaryOperator* param) {
  if (binaryOperators.contains(param)) {
    throw PreviousDeclarationException(param, binaryOperators.get(param));
  } else {
    binaryOperators.add(param);
  }
}

void bi::Scope::add(UnaryOperator* param) {
  if (unaryOperators.contains(param)) {
    throw PreviousDeclarationException(param, unaryOperators.get(param));
  } else {
    unaryOperators.add(param);
  }
}

void bi::Scope::add(AssignmentOperator* param) {
  if (assignmentOperators.contains(param)) {
    throw PreviousDeclarationException(param, assignmentOperators.get(param));
  } else {
    assignmentOperators.add(param);
  }
}

void bi::Scope::add(ConversionOperator* param) {
  if (conversionOperators.contains(param)) {
    throw PreviousDeclarationException(param, conversionOperators.get(param));
  } else {
    conversionOperators.add(param);
  }
}

void bi::Scope::add(TypeParameter* param) {
  if (types.contains(param)) {
    throw PreviousDeclarationException(param, types.get(param));
  } else {
    types.add(param);
  }
}

void bi::Scope::resolve(VarReference* ref) {
  vars.resolve(ref);
  if (ref->matches.size() == 0) {
    resolveDefer(ref);
  }
}

void bi::Scope::resolve(FuncReference* ref) {
  functions.resolve(ref);
  if (ref->matches.size() == 0) {
    resolveDefer(ref);
  }
}

void bi::Scope::resolve(BinaryReference* ref) {
  binaryOperators.resolve(ref);
  if (ref->matches.size() == 0) {
    resolveDefer(ref);
  }
}

void bi::Scope::resolve(UnaryReference* ref) {
  unaryOperators.resolve(ref);
  if (ref->matches.size() == 0) {
    resolveDefer(ref);
  }
}

void bi::Scope::resolve(Assignment* ref) {
  assignmentOperators.resolve(ref);
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
  functions.merge(scope->functions);
  coroutines.merge(scope->coroutines);
  memberFunctions.merge(scope->memberFunctions);
  programs.merge(scope->programs);
  binaryOperators.merge(scope->binaryOperators);
  unaryOperators.merge(scope->unaryOperators);
  assignmentOperators.merge(scope->assignmentOperators);
  conversionOperators.merge(scope->conversionOperators);
  types.merge(scope->types);
}
