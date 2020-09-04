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

birch::TypeParameterised::~TypeParameterised() {
  //
}

bool birch::TypeParameterised::isGeneric() const {
  return !typeParams->isEmpty();
}

birch::Type* birch::TypeParameterised::createArguments() const {
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
