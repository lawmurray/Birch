/**
 * @file
 */
#include "bi/common/TypeParameterised.hpp"

#include "bi/expression/all.hpp"
#include "bi/type/all.hpp"

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

bi::Type* bi::TypeParameterised::createArguments() const {
  Type* typeArgs = nullptr;
  if (typeParams->isEmpty()) {
    typeArgs = new EmptyType();
  } else {
    for (auto typeParam : *typeParams) {
      auto o = dynamic_cast<const Generic*>(typeParam);
      assert(o);
      auto typeArg = new NamedType(o->name, o->loc);
      if (typeArgs) {
        typeArgs = new TypeList(typeArgs, typeArg, typeArg->loc);
      } else {
        typeArgs = typeArg;
      }
    }
  }
  return typeArgs;
}
