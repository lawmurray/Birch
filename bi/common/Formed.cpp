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
  return form == BINARY || form == ASSIGN;
}

bool bi::Formed::isUnary() const {
  return form == UNARY;
}

bool bi::Formed::isAssign() const {
  return form == ASSIGN;
}

bool bi::Formed::isLambda() const {
  return form == LAMBDA;
}
