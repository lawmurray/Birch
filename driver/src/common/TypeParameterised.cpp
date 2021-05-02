/**
 * @file
 */
#include "src/common/TypeParameterised.hpp"

#include "src/expression/all.hpp"
#include "src/type/all.hpp"

birch::TypeParameterised::TypeParameterised(Expression* typeParams) :
    typeParams(typeParams) {
  //
}

bool birch::TypeParameterised::isGeneric() const {
  return !typeParams->isEmpty();
}
