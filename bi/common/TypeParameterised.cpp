/**
 * @file
 */
#include "bi/common/TypeParameterised.hpp"

#include "bi/expression/all.hpp"

bi::TypeParameterised::TypeParameterised(Expression* typeParams) :
    typeParams(typeParams) {
  //
}

bi::TypeParameterised::~TypeParameterised() {
  //
}

bool bi::TypeParameterised::isGeneric() const {
  return !typeParams->isEmpty();
}
