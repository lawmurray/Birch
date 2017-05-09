/**
 * @file
 */
#include "bi/common/Formed.hpp"

bi::Formed::Formed(const SignatureForm form) :
    form(form) {
  //
}

bi::Formed::~Formed() {
  //
}

bool bi::Formed::isBinary() const {
  return form == BINARY_OPERATOR || form == ASSIGN_OPERATOR;
}

bool bi::Formed::isUnary() const {
  return form == UNARY_OPERATOR;
}

bool bi::Formed::isAssign() const {
  return form == ASSIGN_OPERATOR;
}

bool bi::Formed::isLambda() const {
  return form == LAMBDA_FUNCTION;
}

bool bi::Formed::isMember() const {
  return form == MEMBER_FUNCTION;
}
