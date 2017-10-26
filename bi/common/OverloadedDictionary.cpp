/**
 * @file
 */
#include "bi/common/OverloadedDictionary.hpp"

#include "bi/statement/Function.hpp"
#include "bi/statement/Fiber.hpp"
#include "bi/statement/MemberFunction.hpp"
#include "bi/statement/MemberFiber.hpp"
#include "bi/statement/BinaryOperator.hpp"
#include "bi/statement/UnaryOperator.hpp"

template<class ObjectType>
bi::OverloadedDictionary<ObjectType>::~OverloadedDictionary() {
  //
}

template<class ObjectType>
bool bi::OverloadedDictionary<ObjectType>::contains(ObjectType* o) {
  auto iter = objects.find(o->name->str());
  return iter != objects.end() && iter->second->contains(o);
}

template<class ObjectType>
bool bi::OverloadedDictionary<ObjectType>::contains(
    const std::string& name) const {
  return objects.find(name) != objects.end();
}

template<class ObjectType>
ObjectType* bi::OverloadedDictionary<ObjectType>::get(ObjectType* o) {
  /* pre-condition */
  assert(contains(o));

  auto iter = objects.find(o->name->str());
  return dynamic_cast<ObjectType*>(iter->second->get(o));
}

template<class ObjectType>
bi::Overloaded* bi::OverloadedDictionary<ObjectType>::get(
    const std::string& name) {
  /* pre-condition */
  assert(contains(name));

  return objects.find(name)->second;
}

template<class ObjectType>
void bi::OverloadedDictionary<ObjectType>::add(ObjectType* o) {
  assert(!contains(o));
  auto name = o->name->str();
  if (this->contains(name)) {
    this->get(name)->add(o);
  } else {
    objects.insert(std::make_pair(name, new Overloaded(o)));
  }
}

template class bi::OverloadedDictionary<bi::Function>;
template class bi::OverloadedDictionary<bi::Fiber>;
template class bi::OverloadedDictionary<bi::MemberFunction>;
template class bi::OverloadedDictionary<bi::MemberFiber>;
template class bi::OverloadedDictionary<bi::BinaryOperator>;
template class bi::OverloadedDictionary<bi::UnaryOperator>;
