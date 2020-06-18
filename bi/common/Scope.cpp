/**
 * @file
 */
#include "bi/common/Scope.hpp"

#include "bi/expression/Parameter.hpp"
#include "bi/expression/Generic.hpp"
#include "bi/statement/GlobalVariable.hpp"
#include "bi/statement/MemberVariable.hpp"
#include "bi/statement/LocalVariable.hpp"
#include "bi/statement/Function.hpp"
#include "bi/statement/Fiber.hpp"
#include "bi/statement/Program.hpp"
#include "bi/statement/MemberFunction.hpp"
#include "bi/statement/BinaryOperator.hpp"
#include "bi/statement/UnaryOperator.hpp"
#include "bi/statement/AssignmentOperator.hpp"
#include "bi/statement/ConversionOperator.hpp"
#include "bi/statement/Basic.hpp"
#include "bi/statement/Class.hpp"
#include "bi/exception/all.hpp"
#include "bi/visitor/Cloner.hpp"

bi::Scope::Scope(const ScopeCategory category) :
    base(nullptr),
    category(category) {
  //
}

void bi::Scope::lookup(NamedExpression* o) const {
  auto name = o->name->str();

  auto parameter = parameters.find(name);
  auto localVariable = localVariables.find(name);
  auto memberVariable = memberVariables.find(name);
  auto globalVariable = globalVariables.find(name);
  auto memberFunction = memberFunctions.find(name);
  auto function = functions.find(name);
  auto memberFiber = memberFibers.find(name);
  auto fiber = fibers.find(name);
  auto binaryOperator = binaryOperators.find(name);
  auto unaryOperator = unaryOperators.find(name);
  auto program = programs.find(name);

  if (parameter != parameters.end()) {
    o->category = PARAMETER;
    o->number = parameter->second->number;
    o->type = parameter->second->type;
  } else if (localVariable != localVariables.end()) {
    o->category = LOCAL_VARIABLE;
    o->number = localVariable->second->number;
    o->type = localVariable->second->type;
  } else if (memberVariable != memberVariables.end()) {
    o->category = MEMBER_VARIABLE;
    o->number = memberVariable->second->number;
    o->type = memberVariable->second->type;
  } else if (globalVariable != globalVariables.end()) {
    o->category = GLOBAL_VARIABLE;
    o->number = globalVariable->second->number;
    o->type = globalVariable->second->type;
  } else if (memberFunction != memberFunctions.end()) {
    o->category = MEMBER_FUNCTION;
    o->number = memberFunction->second->number;
  } else if (function != functions.end()) {
    o->category = GLOBAL_FUNCTION;
    o->number = function->second->number;
  } else if (memberFiber != memberFibers.end()) {
    o->category = MEMBER_FIBER;
    o->number = memberFiber->second->number;
  } else if (fiber != fibers.end()){
    o->category = GLOBAL_FIBER;
    o->number = fiber->second->number;
  } else if (binaryOperator != binaryOperators.end()) {
    o->category = BINARY_OPERATOR;
    o->number = binaryOperator->second->number;
  } else if (unaryOperator != unaryOperators.end()) {
    o->category = UNARY_OPERATOR;
    o->number = unaryOperator->second->number;
  }
  if (base && !o->category) {
    base->lookup(o);
  }
}

void bi::Scope::lookup(NamedType* o) const {
  auto name = o->name->str();

  auto basicType = basicTypes.find(name);
  auto classType = classTypes.find(name);
  auto genericType = genericTypes.find(name);

  if (basicType != basicTypes.end()) {
    o->category = BASIC_TYPE;
    o->number = basicType->second->number;
  } else if (classType != classTypes.end()) {
    o->category = CLASS_TYPE;
    o->number = classType->second->number;
  } else if (genericType != genericTypes.end()) {
    o->category = GENERIC_TYPE;
    o->number = genericType->second->number;
  }
  //if (base && !o->category) {
  //  base->lookup(o);
  //}
}

void bi::Scope::add(Parameter* o) {
  auto param = parameters.find(o->name->str());
  if (param != parameters.end()) {
    throw RedefinedException(o, param->second);
  }
  auto local = localVariables.find(o->name->str());
  if (local != localVariables.end()) {
    throw RedefinedException(o, local->second);
  }
  parameters.insert(std::make_pair(o->name->str(), o));
}

void bi::Scope::add(LocalVariable* o) {
  auto param = parameters.find(o->name->str());
  if (param != parameters.end()) {
    throw RedefinedException(o, param->second);
  }
  auto local = localVariables.find(o->name->str());
  if (local != localVariables.end()) {
    throw RedefinedException(o, local->second);
  }
  localVariables.insert(std::make_pair(o->name->str(), o));
}

void bi::Scope::add(MemberVariable* o) {
  memberVariables.insert(std::make_pair(o->name->str(), o));
}

void bi::Scope::add(GlobalVariable* o) {
  globalVariables.insert(std::make_pair(o->name->str(), o));
}

void bi::Scope::add(MemberFunction* o) {
  memberFunctions.insert(std::make_pair(o->name->str(), o));
}

void bi::Scope::add(Function* o) {
  functions.insert(std::make_pair(o->name->str(), o));
}

void bi::Scope::add(MemberFiber* o) {
  memberFibers.insert(std::make_pair(o->name->str(), o));
}

void bi::Scope::add(Fiber* o) {
  fibers.insert(std::make_pair(o->name->str(), o));
}

void bi::Scope::add(Program* o) {
  programs.insert(std::make_pair(o->name->str(), o));
}

void bi::Scope::add(BinaryOperator* o) {
  binaryOperators.insert(std::make_pair(o->name->str(), o));
}

void bi::Scope::add(UnaryOperator* o) {
  unaryOperators.insert(std::make_pair(o->name->str(), o));
}

void bi::Scope::add(Basic* o) {
  basicTypes.insert(std::make_pair(o->name->str(), o));
}

void bi::Scope::add(Class* o) {
  classTypes.insert(std::make_pair(o->name->str(), o));
}

void bi::Scope::add(Generic* o) {
  genericTypes.insert(std::make_pair(o->name->str(), o));
}

void bi::Scope::inherit(Class* o) const {
  auto base = dynamic_cast<NamedType*>(o->base);
  if (base) {
    auto name = base->name->str();
    auto iter = classTypes.find(name);
    if (iter != classTypes.end()) {
      o->scope->base = iter->second->scope;

      /* check for loops in inheritance */
      auto scope = o->scope;
      do {
        if (scope->base == o->scope) {
          throw InheritanceLoopException(o);
        }
        scope = scope->base;
      } while (scope);
    }
  }
}

bool bi::Scope::overrides(const std::string& name) const {
  return base && (base->memberFunctions.find(name) != base->memberFunctions.end() ||
      base->memberFibers.find(name) != base->memberFibers.end() ||
      base->overrides(name));
}
