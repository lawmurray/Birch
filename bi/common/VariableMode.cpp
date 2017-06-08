/**
 * @file
 */
#include "bi/common/VariableMode.hpp"

bi::VariableMode::VariableMode(const VariableForm form) :
    form(form) {
  //
}

bi::VariableMode::~VariableMode() {
  //
}

bool bi::VariableMode::isParameter() const {
  return form == PARAMETER;
}

bool bi::VariableMode::isMember() const {
  return form == MEMBER;
}
