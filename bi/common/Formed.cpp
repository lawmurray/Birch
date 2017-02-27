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
  return form == BINARY_OPERATOR || form == ASSIGNMENT_OPERATOR;
}

bool bi::Formed::isUnary() const {
  return form == UNARY_OPERATOR;
}

bool bi::Formed::isAssignment() const {
  return form == ASSIGNMENT_OPERATOR;
}

bool bi::Formed::isConstructor() const {
  return form == CONSTRUCTOR;
}
