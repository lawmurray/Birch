/**
 * @file
 */
#include "bi/common/Scope.hpp"

#include "bi/expression/Identifier.hpp"
#include "bi/expression/Parameter.hpp"
#include "bi/statement/GlobalVariable.hpp"
#include "bi/statement/LocalVariable.hpp"
#include "bi/statement/MemberVariable.hpp"
#include "bi/statement/Function.hpp"
#include "bi/statement/Coroutine.hpp"
#include "bi/statement/Program.hpp"
#include "bi/statement/MemberFunction.hpp"
#include "bi/statement/BinaryOperator.hpp"
#include "bi/statement/UnaryOperator.hpp"
#include "bi/statement/AssignmentOperator.hpp"
#include "bi/statement/ConversionOperator.hpp"
#include "bi/statement/Assignment.hpp"
#include "bi/statement/Class.hpp"
#include "bi/statement/Alias.hpp"
#include "bi/statement/Basic.hpp"
#include "bi/exception/all.hpp"
#include "bi/visitor/Cloner.hpp"

bi::LookupResult bi::Scope::lookup(
    const Identifier<Unknown>* ref) const {
  auto name = ref->name->str();
  if (localVariables.contains(name)) {
    return LOCAL_VARIABLE;
  } else if (parameters.contains(name)) {
    return PARAMETER;
  } else if (memberVariables.contains(name)) {
    return MEMBER_VARIABLE;
  } else if (memberFunctions.contains(name)) {
    return MEMBER_FUNCTION;
  } else if (globalVariables.contains(name)) {
    return GLOBAL_VARIABLE;
  } else if (functions.contains(name)) {
    return FUNCTION;
  } else if (coroutines.contains(name)) {
    return COROUTINE;
  } else {
    return lookupInherit(ref);
  }
}

bi::LookupResult bi::Scope::lookup(const IdentifierType* ref) const {
  auto name = ref->name->str();
  if (basics.contains(name)) {
    return BASIC;
  } else if (classes.contains(name)) {
    return CLASS;
  } else if (aliases.contains(name)) {
    return ALIAS;
  } else {
    return lookupInherit(ref);
  }
}

void bi::Scope::add(Parameter* param) {
  auto name = param->name->str();
  if (parameters.contains(name)) {
    throw PreviousDeclarationException(param, parameters.get(name));
  } else {
    parameters.add(param);
  }
}

void bi::Scope::add(GlobalVariable* param) {
  auto name = param->name->str();
  if (globalVariables.contains(name)) {
    throw PreviousDeclarationException(param, globalVariables.get(name));
  } else if (functions.contains(name)) {
    throw PreviousDeclarationException(param);
  } else if (coroutines.contains(name)) {
    throw PreviousDeclarationException(param);
  } else if (programs.contains(name)) {
    throw PreviousDeclarationException(param);
  } else {
    globalVariables.add(param);
  }
}

void bi::Scope::add(LocalVariable* param) {
  auto name = param->name->str();
  if (localVariables.contains(name)) {
    throw PreviousDeclarationException(param, localVariables.get(name));
  } else if (parameters.contains(name)) {
    throw PreviousDeclarationException(param, parameters.get(name));
  } else {
    localVariables.add(param);
  }
}

void bi::Scope::add(MemberVariable* param) {
  auto name = param->name->str();
  if (memberVariables.contains(name)) {
    throw PreviousDeclarationException(param, memberVariables.get(name));
  } else if (memberFunctions.contains(name)) {
    throw PreviousDeclarationException(param);
  } else {
    memberVariables.add(param);
  }
}

void bi::Scope::add(Function* param) {
  auto name = param->name->str();
  if (coroutines.contains(name)) {
    throw PreviousDeclarationException(param);
  } else if (programs.contains(name)) {
    throw PreviousDeclarationException(param, programs.get(name));
  } else if (globalVariables.contains(name)) {
    throw PreviousDeclarationException(param, globalVariables.get(name));
  } else {
    functions.add(param);
  }
}

void bi::Scope::add(Coroutine* param) {
  auto name = param->name->str();
  if (functions.contains(name)) {
    throw PreviousDeclarationException(param);
  } else if (programs.contains(name)) {
    throw PreviousDeclarationException(param, programs.get(name));
  } else if (globalVariables.contains(name)) {
    throw PreviousDeclarationException(param, globalVariables.get(name));
  } else {
    coroutines.add(param);
  }
}

void bi::Scope::add(Program* param) {
  auto name = param->name->str();
  if (functions.contains(name)) {
    throw PreviousDeclarationException(param);
  } else if (coroutines.contains(name)) {
    throw PreviousDeclarationException(param);
  } else if (globalVariables.contains(name)) {
    throw PreviousDeclarationException(param, globalVariables.get(name));
  } else {
    programs.add(param);
  }
}

void bi::Scope::add(MemberFunction* param) {
  auto name = param->name->str();
  if (memberVariables.contains(name)) {
    throw PreviousDeclarationException(param, memberVariables.get(name));
  } else {
    memberFunctions.add(param);
  }
}

void bi::Scope::add(BinaryOperator* param) {
  binaryOperators.add(param);
}

void bi::Scope::add(UnaryOperator* param) {
  unaryOperators.add(param);
}

void bi::Scope::add(Basic* param) {
  auto name = param->name->str();
  if (basics.contains(name)) {
    throw PreviousDeclarationException(param, basics.get(name));
  } else if (classes.contains(name)) {
    throw PreviousDeclarationException(param, classes.get(name));
  } else if (aliases.contains(name)) {
    throw PreviousDeclarationException(param, aliases.get(name));
  } else {
    basics.add(param);
  }
}

void bi::Scope::add(Class* param) {
  auto name = param->name->str();
  if (basics.contains(name)) {
    throw PreviousDeclarationException(param, basics.get(name));
  } else if (classes.contains(name)) {
    throw PreviousDeclarationException(param, classes.get(name));
  } else if (aliases.contains(name)) {
    throw PreviousDeclarationException(param, aliases.get(name));
  } else {
    classes.add(param);
  }
}

void bi::Scope::add(Alias* param) {
  auto name = param->name->str();
  if (basics.contains(name)) {
    throw PreviousDeclarationException(param, basics.get(name));
  } else if (classes.contains(name)) {
    throw PreviousDeclarationException(param, classes.get(name));
  } else if (aliases.contains(name)) {
    throw PreviousDeclarationException(param, aliases.get(name));
  } else {
    aliases.add(param);
  }
}

#define RESOLVE_IMPL(type, container) \
  void bi::Scope::resolve(Identifier<type>* ref) { \
    container.resolve(ref); \
    if (ref->matches.size() == 0) { \
      resolveInherit(ref); \
    } \
  }

RESOLVE_IMPL(Parameter, parameters)
RESOLVE_IMPL(GlobalVariable, globalVariables)
RESOLVE_IMPL(LocalVariable, localVariables)
RESOLVE_IMPL(MemberVariable, memberVariables)
RESOLVE_IMPL(Function, functions)
RESOLVE_IMPL(Coroutine, coroutines)
RESOLVE_IMPL(MemberFunction, memberFunctions)
RESOLVE_IMPL(BinaryOperator, binaryOperators)
RESOLVE_IMPL(UnaryOperator, unaryOperators)

#define RESOLVE_TYPE_IMPL(type, container) \
  void bi::Scope::resolve(type* ref) { \
    container.resolve(ref); \
    if (ref->matches.size() == 0) { \
      resolveInherit(ref); \
    } \
  }

RESOLVE_TYPE_IMPL(BasicType, basics)
RESOLVE_TYPE_IMPL(ClassType, classes)
RESOLVE_TYPE_IMPL(AliasType, aliases)

void bi::Scope::inherit(Scope* scope) {
  bases.insert(scope);
}

void bi::Scope::import(Scope* scope) {
  // only file-scope objects need to be imported here, e.g. no need for
  // class members
  globalVariables.import(scope->globalVariables);
  functions.import(scope->functions);
  coroutines.import(scope->coroutines);
  programs.import(scope->programs);
  binaryOperators.import(scope->binaryOperators);
  unaryOperators.import(scope->unaryOperators);
  basics.import(scope->basics);
  classes.import(scope->classes);
  aliases.import(scope->aliases);
}
