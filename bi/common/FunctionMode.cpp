/**
 * @file
 */
#include "bi/common/FunctionMode.hpp"

bi::FunctionMode::FunctionMode(const FunctionForm form) :
    form(form) {
  //
}

bi::FunctionMode::~FunctionMode() {
  //
}

bool bi::FunctionMode::isOperator() const {
  return isBinary() || isUnary() || isAssign();
}

bool bi::FunctionMode::isBinary() const {
  return form == BINARY_FORM || form == ASSIGN_FORM;
}

bool bi::FunctionMode::isUnary() const {
  return form == UNARY_FORM;
}

bool bi::FunctionMode::isAssign() const {
  return form == ASSIGN_FORM;
}

bool bi::FunctionMode::isLambda() const {
  return form == LAMBDA_FORM;
}

bool bi::FunctionMode::isMember() const {
  return form == MEMBER_FUNCTION_FORM;
}

bool bi::FunctionMode::isCoroutine() const {
  return form == COROUTINE_FORM;
}
