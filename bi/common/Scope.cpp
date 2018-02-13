/**
 * @file
 */
#include "bi/common/Scope.hpp"

#include "bi/expression/Identifier.hpp"
#include "bi/expression/OverloadedIdentifier.hpp"
#include "bi/expression/LocalVariable.hpp"
#include "bi/expression/Parameter.hpp"
#include "bi/expression/MemberParameter.hpp"
#include "bi/expression/Generic.hpp"
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
#include "bi/statement/Basic.hpp"
#include "bi/statement/Class.hpp"
#include "bi/exception/all.hpp"
#include "bi/visitor/Cloner.hpp"

bi::Scope::Scope(const ScopeCategory category) :
    base(nullptr),
    category(category) {
  //
}

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
  } else if (base) {
    return base->lookupInherit(ref);
  } else {
    return UNRESOLVED;
  }
}

bi::LookupResult bi::Scope::lookup(const UnknownType* ref) const {
  auto name = ref->name->str();
  if (basics.contains(name)) {
    return BASIC;
  } else if (classes.contains(name)) {
    return CLASS;
  } else if (generics.contains(name)) {
    return GENERIC;
  } else if (base) {
    return base->lookupInherit(ref);
  } else {
    return UNRESOLVED;
  }
}

bi::LookupResult bi::Scope::lookupInherit(
    const Identifier<Unknown>* ref) const {
  auto name = ref->name->str();
  if (memberVariables.contains(name)) {
    return MEMBER_VARIABLE;
  } else if (memberParameters.contains(name)) {
    return MEMBER_PARAMETER;
  } else if (memberFunctions.contains(name)) {
    return MEMBER_FUNCTION;
  } else if (memberFibers.contains(name)) {
    return MEMBER_FIBER;
  } else if (base) {
    return base->lookupInherit(ref);
  } else {
    return UNRESOLVED;
  }
}

bi::LookupResult bi::Scope::lookupInherit(const UnknownType* ref) const {
  auto name = ref->name->str();
  if (basics.contains(name)) {
    return BASIC;
  } else if (classes.contains(name)) {
    return CLASS;
  } else if (generics.contains(name)) {
    return GENERIC;
  } else if (base) {
    return base->lookupInherit(ref);
  } else {
    return UNRESOLVED;
  }
}

void bi::Scope::add(Parameter* param) {
  checkPreviousLocal(param);
  parameters.add(param);
}

void bi::Scope::add(MemberParameter* param) {
  checkPreviousMember(param);
  memberParameters.add(param);
}

void bi::Scope::add(GlobalVariable* param) {
  checkPreviousGlobal(param);
  globalVariables.add(param);
}

void bi::Scope::add(LocalVariable* param) {
  checkPreviousLocal(param);
  localVariables.add(param);
}

void bi::Scope::add(MemberVariable* param) {
  checkPreviousMember(param);
  memberVariables.add(param);
}

void bi::Scope::add(Function* param) {
  checkPreviousGlobal(param);
  if (functions.contains(param)) {
    throw PreviousDeclarationException(param, functions.get(param));
  } else if (fibers.contains(param->name->str())) {
    throw PreviousDeclarationException(param);
  }
  functions.add(param);
}

void bi::Scope::add(Fiber* param) {
  checkPreviousGlobal(param);
  if (functions.contains(param->name->str())) {
    throw PreviousDeclarationException(param);
  }
  if (fibers.contains(param)) {
    throw PreviousDeclarationException(param, fibers.get(param));
  }
  fibers.add(param);
}

void bi::Scope::add(Program* param) {
  checkPreviousGlobal(param);
  programs.add(param);
}

void bi::Scope::add(MemberFunction* param) {
  checkPreviousMember(param);
  if (memberFunctions.contains(param)
      || memberFibers.contains(param->name->str())) {
    throw PreviousDeclarationException(param);
  }
  memberFunctions.add(param);
}

void bi::Scope::add(MemberFiber* param) {
  checkPreviousMember(param);
  if (memberFunctions.contains(param->name->str())
      || memberFibers.contains(param)) {
    throw PreviousDeclarationException(param);
  }
  memberFibers.add(param);
}

void bi::Scope::add(BinaryOperator* param) {
  binaryOperators.add(param);
}

void bi::Scope::add(UnaryOperator* param) {
  unaryOperators.add(param);
}

void bi::Scope::add(Basic* param) {
  checkPreviousType(param);
  basics.add(param);
}

void bi::Scope::add(Class* param) {
  checkPreviousType(param);
  classes.add(param);
}

void bi::Scope::add(Generic* param) {
  checkPreviousType(param);
  generics.add(param);
}

void bi::Scope::resolve(Identifier<Parameter>* ref) {
  parameters.resolve(ref);
}

void bi::Scope::resolve(Identifier<MemberParameter>* ref) {
  memberParameters.resolve(ref);
  if (!ref->target && base) {
    base->resolve(ref);
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
  if (!ref->target && base) {
    base->resolve(ref);
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
  if (!ref->target && base) {
    base->resolve(ref);
  }
}

void bi::Scope::resolve(OverloadedIdentifier<MemberFiber>* ref) {
  memberFibers.resolve(ref);
  if (!ref->target && base) {
    base->resolve(ref);
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

void bi::Scope::resolve(GenericType* ref) {
  generics.resolve(ref);
  if (!ref->target && base) {
    base->resolve(ref);
  }
}

void bi::Scope::inherit(Scope* scope) {
  assert(!base);
  base = scope;
}

template<class ParameterType>
void bi::Scope::checkPreviousGlobal(ParameterType* param) {
  auto name = param->name->str();
  if (programs.contains(name)) {
    throw PreviousDeclarationException(param, programs.get(name));
  } else if (globalVariables.contains(name)) {
    throw PreviousDeclarationException(param, globalVariables.get(name));
  }
}

template<class ParameterType>
void bi::Scope::checkPreviousLocal(ParameterType* param) {
  auto name = param->name->str();
  if (localVariables.contains(name)) {
    throw PreviousDeclarationException(param, localVariables.get(name));
  } else if (parameters.contains(name)) {
    throw PreviousDeclarationException(param, parameters.get(name));
  }
}

template<class ParameterType>
void bi::Scope::checkPreviousMember(ParameterType* param) {
  auto name = param->name->str();
  if (memberVariables.contains(name)) {
    throw PreviousDeclarationException(param, memberVariables.get(name));
  } else if (memberParameters.contains(name)) {
    throw PreviousDeclarationException(param, memberParameters.get(name));
  }
}

template<class ParameterType>
void bi::Scope::checkPreviousType(ParameterType* param) {
  auto name = param->name->str();
  if (basics.contains(name)) {
    throw PreviousDeclarationException(param, basics.get(name));
  } else if (classes.contains(name)) {
    throw PreviousDeclarationException(param, classes.get(name));
  } else if (generics.contains(name)) {
    throw PreviousDeclarationException(param, generics.get(name));
  }
}
