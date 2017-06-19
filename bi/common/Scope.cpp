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

#define CONTAINS_IMPL(type, container) \
  bool bi::Scope::contains(type* param) { \
    return container.contains(param); \
  }

CONTAINS_IMPL(Parameter, parameters)
CONTAINS_IMPL(GlobalVariable, globalVariables)
CONTAINS_IMPL(LocalVariable, localVariables)
CONTAINS_IMPL(MemberVariable, memberVariables)
CONTAINS_IMPL(Function, functions)
CONTAINS_IMPL(Coroutine, coroutines)
CONTAINS_IMPL(Program, programs)
CONTAINS_IMPL(MemberFunction, memberFunctions)
CONTAINS_IMPL(BinaryOperator, binaryOperators)
CONTAINS_IMPL(UnaryOperator, unaryOperators)
CONTAINS_IMPL(AssignmentOperator, assignmentOperators)
CONTAINS_IMPL(ConversionOperator, conversionOperators)
CONTAINS_IMPL(Basic, basics)
CONTAINS_IMPL(Class, classes)
CONTAINS_IMPL(Alias, aliases)

#define ADD_IMPL(type, container) \
  void bi::Scope::add(type* param) { \
    if (container.contains(param)) { \
      throw PreviousDeclarationException(param, container.get(param)); \
    } else { \
      container.add(param); \
    } \
  }

ADD_IMPL(Parameter, parameters)
ADD_IMPL(GlobalVariable, globalVariables)
ADD_IMPL(LocalVariable, localVariables)
ADD_IMPL(MemberVariable, memberVariables)
ADD_IMPL(Function, functions)
ADD_IMPL(Coroutine, coroutines)
ADD_IMPL(Program, programs)
ADD_IMPL(MemberFunction, memberFunctions)
ADD_IMPL(BinaryOperator, binaryOperators)
ADD_IMPL(UnaryOperator, unaryOperators)
ADD_IMPL(AssignmentOperator, assignmentOperators)
ADD_IMPL(ConversionOperator, conversionOperators)
ADD_IMPL(Basic, basics)
ADD_IMPL(Class, classes)
ADD_IMPL(Alias, aliases)

#define RESOLVE_IMPL(type, container) \
  void bi::Scope::resolve(Identifier<type>* ref) { \
    container.resolve(ref); \
    if (ref->matches.size() == 0) { \
      resolveDefer(ref); \
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
      resolveDefer(ref); \
    } \
  }

RESOLVE_TYPE_IMPL(BasicType, basics)
RESOLVE_TYPE_IMPL(ClassType, classes)
RESOLVE_TYPE_IMPL(AliasType, aliases)

void bi::Scope::resolve(Assignment* ref) {
  assignmentOperators.resolve(ref);
  if (ref->matches.size() == 0) {
    resolveDefer(ref);
  }
}

void bi::Scope::inherit(Scope* scope) {
  bases.insert(scope);
}

void bi::Scope::import(Scope* scope) {
  parameters.merge(scope->parameters);
  globalVariables.merge(scope->globalVariables);
  localVariables.merge(scope->localVariables);
  memberVariables.merge(scope->memberVariables);
  functions.merge(scope->functions);
  coroutines.merge(scope->coroutines);
  memberFunctions.merge(scope->memberFunctions);
  programs.merge(scope->programs);
  binaryOperators.merge(scope->binaryOperators);
  unaryOperators.merge(scope->unaryOperators);
  assignmentOperators.merge(scope->assignmentOperators);
  conversionOperators.merge(scope->conversionOperators);
  basics.merge(scope->basics);
  classes.merge(scope->classes);
  aliases.merge(scope->aliases);
}
