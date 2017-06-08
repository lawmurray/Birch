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
  return form == BINARY_OPERATOR || form == ASSIGN_OPERATOR;
}

bool bi::FunctionMode::isUnary() const {
  return form == UNARY_OPERATOR;
}

bool bi::FunctionMode::isAssign() const {
  return form == ASSIGN_OPERATOR;
}

bool bi::FunctionMode::isLambda() const {
  return form == LAMBDA_FUNCTION;
}

bool bi::FunctionMode::isMember() const {
  return form == MEMBER_FUNCTION;
}
