/**
 * @file
 */
#include "bi/common/Scope.hpp"

#include "bi/expression/Identifier.hpp"
#include "bi/expression/OverloadedIdentifier.hpp"
#include "bi/expression/Parameter.hpp"
#include "bi/expression/MemberParameter.hpp"
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

bi::LookupResult bi::Scope::lookup(const Identifier<Unknown>* ref) const {
  auto name = ref->name->str();
  if (localVariables.contains(name)) {
    return LOCAL_VARIABLE;
  } else if (parameters.contains(name)) {
    return PARAMETER;
  } else if (memberVariables.contains(name)) {
    return MEMBER_VARIABLE;
  } else if (memberParameters.contains(name)) {
    return MEMBER_PARAMETER;
  } else if (memberFunctions.contains(name)) {
    return MEMBER_FUNCTION;
  } else if (memberCoroutines.contains(name)) {
    return MEMBER_COROUTINE;
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

void bi::Scope::add(MemberParameter* param) {
  auto name = param->name->str();
  if (memberVariables.contains(name)) {
    throw PreviousDeclarationException(param, memberVariables.get(name));
  } else if (memberParameters.contains(name)) {
    throw PreviousDeclarationException(param, memberParameters.get(name));
  } else if (memberFunctions.contains(name)) {
    throw PreviousDeclarationException(param);
  } else {
    memberParameters.add(param);
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
  } else if (memberParameters.contains(name)) {
    throw PreviousDeclarationException(param, memberParameters.get(name));
  } else if (memberFunctions.contains(name)) {
    throw PreviousDeclarationException(param);
  } else if (memberCoroutines.contains(name)) {
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
  } else if (memberParameters.contains(name)) {
    throw PreviousDeclarationException(param, memberParameters.get(name));
  } else if (memberCoroutines.contains(name)) {
    throw PreviousDeclarationException(param);
  } else {
    memberFunctions.add(param);
  }
}

void bi::Scope::add(MemberCoroutine* param) {
  auto name = param->name->str();
  if (memberVariables.contains(name)) {
    throw PreviousDeclarationException(param, memberVariables.get(name));
  } else if (memberParameters.contains(name)) {
    throw PreviousDeclarationException(param, memberParameters.get(name));
  } else if (memberFunctions.contains(name)) {
    throw PreviousDeclarationException(param);
  } else {
    memberCoroutines.add(param);
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

void bi::Scope::resolve(Identifier<Parameter>* ref) {
  parameters.resolve(ref);
}

void bi::Scope::resolve(Identifier<MemberParameter>* ref) {
  memberParameters.resolve(ref);
  /* member parameters are used by constructors, and are deliberately not
   * inherited from base classes */
}

void bi::Scope::resolve(Identifier<GlobalVariable>* ref) {
  globalVariables.resolve(ref);
}

void bi::Scope::resolve(Identifier<LocalVariable>* ref) {
  localVariables.resolve(ref);
}

void bi::Scope::resolve(Identifier<MemberVariable>* ref) {
  memberVariables.resolve(ref);
  if (ref->matches.size() == 0) {
    resolveInherit(ref);
  }
}

void bi::Scope::resolve(OverloadedIdentifier<Function>* ref) {
  functions.resolve(ref);
}

void bi::Scope::resolve(OverloadedIdentifier<Coroutine>* ref) {
  coroutines.resolve(ref);
}

void bi::Scope::resolve(OverloadedIdentifier<MemberFunction>* ref) {
  memberFunctions.resolve(ref);
  if (ref->matches.size() == 0) {
    resolveInherit(ref);
  }
}

void bi::Scope::resolve(OverloadedIdentifier<MemberCoroutine>* ref) {
  memberCoroutines.resolve(ref);
  if (ref->matches.size() == 0) {
    resolveInherit(ref);
  }
}

void bi::Scope::resolve(OverloadedIdentifier<BinaryOperator>* ref) {
  binaryOperators.resolve(ref);
}

void bi::Scope::resolve(OverloadedIdentifier<UnaryOperator>* ref) {
  unaryOperators.resolve(ref);
}

void bi::Scope::resolve(BasicType* ref) {
  basics.resolve(ref);
}

void bi::Scope::resolve(ClassType* ref) {
  classes.resolve(ref);
}

void bi::Scope::resolve(AliasType* ref) {
  aliases.resolve(ref);
}

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
