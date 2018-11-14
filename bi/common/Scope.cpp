/**
 * @file
 */
#include "bi/common/Scope.hpp"

#include "bi/expression/Identifier.hpp"
#include "bi/expression/OverloadedIdentifier.hpp"
#include "bi/expression/LocalVariable.hpp"
#include "bi/expression/Parameter.hpp"
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

bi::LookupResult bi::Scope::lookup(const Identifier<Unknown>* o) const {
  auto name = o->name->str();
  if (localVariables.contains(name)) {
    return LOCAL_VARIABLE;
  } else if (parameters.contains(name)) {
    return PARAMETER;
  } else if (memberVariables.contains(name)) {
    return MEMBER_VARIABLE;
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
    return base->lookupInherit(o);
  } else {
    return UNRESOLVED;
  }
}

bi::LookupResult bi::Scope::lookup(const OverloadedIdentifier<Unknown>* o) const {
  auto name = o->name->str();
  if (memberFunctions.contains(name)) {
    return MEMBER_FUNCTION;
  } else if (memberFibers.contains(name)) {
    return MEMBER_FIBER;
  } else if (functions.contains(name)) {
    return FUNCTION;
  } else if (fibers.contains(name)) {
    return FIBER;
  } else if (base) {
    return base->lookupInherit(o);
  } else {
    return UNRESOLVED;
  }
}

bi::LookupResult bi::Scope::lookup(const UnknownType* o) const {
  auto name = o->name->str();
  if (basics.contains(name)) {
    return BASIC;
  } else if (classes.contains(name)) {
    return CLASS;
  } else if (generics.contains(name)) {
    return GENERIC;
  } else if (base) {
    return base->lookupInherit(o);
  } else {
    return UNRESOLVED;
  }
}

bi::LookupResult bi::Scope::lookupInherit(
    const Identifier<Unknown>* o) const {
  auto name = o->name->str();
  if (memberVariables.contains(name)) {
    return MEMBER_VARIABLE;
  } else if (memberFunctions.contains(name)) {
    return MEMBER_FUNCTION;
  } else if (memberFibers.contains(name)) {
    return MEMBER_FIBER;
  } else if (base) {
    return base->lookupInherit(o);
  } else {
    return UNRESOLVED;
  }
}

bi::LookupResult bi::Scope::lookupInherit(
    const OverloadedIdentifier<Unknown>* o) const {
  auto name = o->name->str();
  if (memberFunctions.contains(name)) {
    return MEMBER_FUNCTION;
  } else if (memberFibers.contains(name)) {
    return MEMBER_FIBER;
  } else if (base) {
    return base->lookupInherit(o);
  } else {
    return UNRESOLVED;
  }
}

bi::LookupResult bi::Scope::lookupInherit(const UnknownType* o) const {
  auto name = o->name->str();
  if (basics.contains(name)) {
    return BASIC;
  } else if (classes.contains(name)) {
    return CLASS;
  } else if (generics.contains(name)) {
    return GENERIC;
  } else if (base) {
    return base->lookupInherit(o);
  } else {
    return UNRESOLVED;
  }
}

void bi::Scope::add(Parameter* param) {
  checkPreviousLocal(param);
  parameters.add(param);
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
  if (binaryOperators.contains(param)) {
    throw PreviousDeclarationException(param);
  }
  binaryOperators.add(param);
}

void bi::Scope::add(UnaryOperator* param) {
  if (unaryOperators.contains(param)) {
    throw PreviousDeclarationException(param);
  }
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

void bi::Scope::resolve(Identifier<Parameter>* o) {
  parameters.resolve(o);
}

void bi::Scope::resolve(Identifier<GlobalVariable>* o) {
  globalVariables.resolve(o);
}

void bi::Scope::resolve(Identifier<LocalVariable>* o) {
  localVariables.resolve(o);
}

void bi::Scope::resolve(Identifier<MemberVariable>* o) {
  memberVariables.resolve(o);
  if (!o->target && base) {
    base->resolve(o);
  }
}

void bi::Scope::resolve(OverloadedIdentifier<Function>* o) {
  functions.resolve(o);
}

void bi::Scope::resolve(OverloadedIdentifier<Fiber>* o) {
  fibers.resolve(o);
}

void bi::Scope::resolve(OverloadedIdentifier<MemberFunction>* o) {
  memberFunctions.resolve(o);
  if (base) {
    /* gather alternatives from base classes */
    base->resolve(o);
  }
}

void bi::Scope::resolve(OverloadedIdentifier<MemberFiber>* o) {
  memberFibers.resolve(o);
  if (base) {
    /* gather alternatives from base classes */
    base->resolve(o);
  }
}

void bi::Scope::resolve(OverloadedIdentifier<BinaryOperator>* o) {
  binaryOperators.resolve(o);
}

void bi::Scope::resolve(OverloadedIdentifier<UnaryOperator>* o) {
  unaryOperators.resolve(o);
}

void bi::Scope::resolve(BasicType* o) {
  basics.resolve(o);
}

void bi::Scope::resolve(ClassType* o) {
  classes.resolve(o);
}

void bi::Scope::resolve(GenericType* o) {
  generics.resolve(o);
  if (!o->target && base) {
    base->resolve(o);
  }
}

void bi::Scope::inherit(Scope* scope) {
  assert(scope);
  assert(!base);
  base = scope;
}

bool bi::Scope::override(const MemberFunction* o) const {
  return base && (base->memberFunctions.contains(o->name->str()) || base->override(o));
}

bool bi::Scope::override(const MemberFiber* o) const {
  return base && (base->memberFibers.contains(o->name->str()) || base->override(o));
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
    assert(false);
    throw PreviousDeclarationException(param, generics.get(name));
  }
}
