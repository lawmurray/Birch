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
  return isAssign();
}

bool bi::FunctionMode::isAssign() const {
  return form == ASSIGN_FORM;
}

bool bi::FunctionMode::isLambda() const {
  return form == LAMBDA_FUNCTION_FORM || form == LAMBDA_COROUTINE_FORM;
}

bool bi::FunctionMode::isMember() const {
  return form == MEMBER_FUNCTION_FORM || form == MEMBER_COROUTINE_FORM;
}

bool bi::FunctionMode::isFunction() const {
  return form == FUNCTION_FORM || form == MEMBER_FUNCTION_FORM || form == LAMBDA_FUNCTION_FORM;
}

bool bi::FunctionMode::isCoroutine() const {
  return form == COROUTINE_FORM || form == MEMBER_COROUTINE_FORM || form == LAMBDA_COROUTINE_FORM;
}
