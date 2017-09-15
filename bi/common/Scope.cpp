/**
 * @file
 */
#include "bi/common/Scope.hpp"

#include "bi/expression/Identifier.hpp"
#include "bi/expression/OverloadedIdentifier.hpp"
#include "bi/expression/LocalVariable.hpp"
#include "bi/expression/Parameter.hpp"
#include "bi/expression/MemberParameter.hpp"
#include "bi/statement/GlobalVariable.hpp"
#include "bi/statement/MemberVariable.hpp"
#include "bi/statement/Function.hpp"
#include "bi/statement/Fiber.hpp"
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
  } else if (memberFibers.contains(name)) {
    return MEMBER_FIBER;
  } else if (globalVariables.contains(name)) {
    return GLOBAL_VARIABLE;
  } else if (functions.contains(name)) {
    return FUNCTION;
  } else if (fibers.contains(name)) {
    return FIBER;
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
  } else if (fibers.contains(name)) {
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
  } else if (memberFibers.contains(name)) {
    throw PreviousDeclarationException(param);
  } else {
    memberVariables.add(param);
  }
}

void bi::Scope::add(Function* param) {
  auto name = param->name->str();
  if (functions.contains(param)) {
    throw PreviousDeclarationException(param);
  } else if (fibers.contains(name)) {
    throw PreviousDeclarationException(param);
  } else if (programs.contains(name)) {
    throw PreviousDeclarationException(param, programs.get(name));
  } else if (globalVariables.contains(name)) {
    throw PreviousDeclarationException(param, globalVariables.get(name));
  } else {
    functions.add(param);
  }
}

void bi::Scope::add(Fiber* param) {
  auto name = param->name->str();
  if (fibers.contains(param)) {
    throw PreviousDeclarationException(param);
  } else if (functions.contains(name)) {
    throw PreviousDeclarationException(param);
  } else if (programs.contains(name)) {
    throw PreviousDeclarationException(param, programs.get(name));
  } else if (globalVariables.contains(name)) {
    throw PreviousDeclarationException(param, globalVariables.get(name));
  } else {
    fibers.add(param);
  }
}

void bi::Scope::add(Program* param) {
  auto name = param->name->str();
  if (programs.contains(name)) {
    throw PreviousDeclarationException(param, programs.get(name));
  } else if (functions.contains(name)) {
    throw PreviousDeclarationException(param);
  } else if (fibers.contains(name)) {
    throw PreviousDeclarationException(param);
  } else if (globalVariables.contains(name)) {
    throw PreviousDeclarationException(param, globalVariables.get(name));
  } else {
    programs.add(param);
  }
}

void bi::Scope::add(MemberFunction* param) {
  auto name = param->name->str();
  if (memberFunctions.contains(param)) {
    throw PreviousDeclarationException(param);
  } else if (memberVariables.contains(name)) {
    throw PreviousDeclarationException(param, memberVariables.get(name));
  } else if (memberParameters.contains(name)) {
    throw PreviousDeclarationException(param, memberParameters.get(name));
  } else if (memberFibers.contains(name)) {
    throw PreviousDeclarationException(param);
  } else {
    memberFunctions.add(param);
  }
}

void bi::Scope::add(MemberFiber* param) {
  auto name = param->name->str();
  if (memberFibers.contains(param)) {
    throw PreviousDeclarationException(param);
  } else if (memberVariables.contains(name)) {
    throw PreviousDeclarationException(param, memberVariables.get(name));
  } else if (memberParameters.contains(name)) {
    throw PreviousDeclarationException(param, memberParameters.get(name));
  } else if (memberFunctions.contains(name)) {
    throw PreviousDeclarationException(param);
  } else {
    memberFibers.add(param);
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
  if (!ref->target) {
    resolveInherit(ref);
  }
}

void bi::Scope::resolve(Identifier<GlobalVariable>* ref) {
  globalVariables.resolve(ref);
}

void bi::Scope::resolve(Identifier<LocalVariable>* ref) {
  localVariables.resolve(ref);
}

void bi::Scope::resolve(Identifier<MemberVariable>* ref) {
  memberVariables.resolve(ref);
  if (!ref->target) {
    resolveInherit(ref);
  }
}

void bi::Scope::resolve(OverloadedIdentifier<Function>* ref) {
  functions.resolve(ref);
}

void bi::Scope::resolve(OverloadedIdentifier<Fiber>* ref) {
  fibers.resolve(ref);
}

void bi::Scope::resolve(OverloadedIdentifier<MemberFunction>* ref) {
  memberFunctions.resolve(ref);
  if (!ref->target) {
    resolveInherit(ref);
  }
}

void bi::Scope::resolve(OverloadedIdentifier<MemberFiber>* ref) {
  memberFibers.resolve(ref);
  if (!ref->target) {
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
