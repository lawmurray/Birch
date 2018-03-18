/**
 * @file
 */
#include "bi/visitor/Instantiater.hpp"

bi::Instantiater::Instantiater(Type* typeArgs) : iter(typeArgs->begin()) {
  //
}

bi::Expression* bi::Instantiater::clone(const Generic* o) {
  auto type = (*(iter++))->canonical()->accept(this);
  map.insert(std::make_pair(o->name->str(), type));
  return new Generic(o->name, type, o->loc);
}

bi::Type* bi::Instantiater::clone(const GenericType* o) {
  auto iter = map.find(o->name->str());
  if (iter != map.end()) {
    return iter->second->accept(this);
  } else {
    return Cloner::clone(o);
  }
}

bi::Type* bi::Instantiater::clone(const UnknownType* o) {
  auto iter = map.find(o->name->str());
  if (iter != map.end()) {
    return iter->second->accept(this);
  } else {
    return Cloner::clone(o);
  }
}
