/**
 * @file
 */
#include "bi/visitor/Instantiater.hpp"

bi::Instantiater::Instantiater(const Statement* typeParams, const Type* typeArgs) {
  auto param = typeParams->begin();
  auto arg = typeArgs->begin();
  while (param != typeParams->end() && arg != typeArgs->end()) {
    auto generic = dynamic_cast<const Generic*>(*param);
    assert(generic);
    substitutions.insert(std::make_pair(generic->name->str(), *arg));
    ++param;
    ++arg;
  }
}

bi::Instantiater::~Instantiater() {
  //
}
