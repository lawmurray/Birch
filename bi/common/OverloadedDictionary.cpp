/**
 * @file
 */
#include "bi/common/OverloadedDictionary.hpp"

#include "bi/statement/Function.hpp"
#include "bi/statement/Coroutine.hpp"
#include "bi/statement/MemberFunction.hpp"
#include "bi/statement/MemberCoroutine.hpp"
#include "bi/statement/BinaryOperator.hpp"
#include "bi/statement/UnaryOperator.hpp"

template<class ObjectType>
bi::OverloadedDictionary<ObjectType>::~OverloadedDictionary() {
  for (auto o : this->objects) {
    delete o.second;
  }
  this->objects.clear();
}

template<class ObjectType>
bool bi::OverloadedDictionary<ObjectType>::contains(ObjectType* o) const {
  auto iter = objects.find(o->name->str());
  return iter != objects.end() && iter->second->contains(o);
}

template<class ObjectType>
bool bi::OverloadedDictionary<ObjectType>::contains(
    const std::string& name) const {
  return objects.find(name) != objects.end();
}

template<class ObjectType>
bi::Overloaded<ObjectType>* bi::OverloadedDictionary<ObjectType>::get(
    const std::string& name) {
  /* pre-condition */
  assert(contains(name));

  return objects.find(name)->second;
}

template<class ObjectType>
void bi::OverloadedDictionary<ObjectType>::add(ObjectType* o) {
  if (this->contains(o->name->str())) {
    this->get(o->name->str())->add(o);
  } else {
    objects.insert(
        std::make_pair(o->name->str(), new Overloaded<ObjectType>(o)));
  }
}

template<class ObjectType>
void bi::OverloadedDictionary<ObjectType>::import(
    OverloadedDictionary<ObjectType>& o) {
  for (auto object : o.objects) {
    for (auto overload : object.second->overloads) {
      if (!contains(overload)) {
        add(overload);
      }
    }
  }
}

template class bi::OverloadedDictionary<bi::Function>;
template class bi::OverloadedDictionary<bi::Coroutine>;
template class bi::OverloadedDictionary<bi::MemberFunction>;
template class bi::OverloadedDictionary<bi::MemberCoroutine>;
template class bi::OverloadedDictionary<bi::BinaryOperator>;
template class bi::OverloadedDictionary<bi::UnaryOperator>;
