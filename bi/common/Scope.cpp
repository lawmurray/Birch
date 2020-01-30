/**
#include <bi/expression/NamedExpression.hpp>
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

bi::Scope::Scope(const ScopeCategory category) : category(category) {
  //
}

bool bi::Scope::lookup(const NamedExpression* o) const {
  return names.find(o->name->str()) != names.end();
}

bool bi::Scope::lookup(const NamedType* o) const {
  return typeNames.find(o->name->str()) != names.end();
}

void bi::Scope::add(Parameter* param) {
  names.insert(param->name->str());
}

void bi::Scope::add(GlobalVariable* param) {
  names.insert(param->name->str());
}

void bi::Scope::add(MemberVariable* param) {
  names.insert(param->name->str());
}

void bi::Scope::add(LocalVariable* param) {
  names.insert(param->name->str());
}

void bi::Scope::add(Function* param) {
  names.insert(param->name->str());
}

void bi::Scope::add(Fiber* param) {
  names.insert(param->name->str());
}

void bi::Scope::add(Program* param) {
  names.insert(param->name->str());
}

void bi::Scope::add(MemberFunction* param) {
  names.insert(param->name->str());
}

void bi::Scope::add(MemberFiber* param) {
  names.insert(param->name->str());
}

void bi::Scope::add(BinaryOperator* param) {
  names.insert(param->name->str());
}

void bi::Scope::add(UnaryOperator* param) {
  names.insert(param->name->str());
}

void bi::Scope::add(Basic* param) {
  typeNames.insert(param->name->str());
}

void bi::Scope::add(Class* param) {
  typeNames.insert(param->name->str());
}

void bi::Scope::add(Generic* param) {
  typeNames.insert(param->name->str());
}
