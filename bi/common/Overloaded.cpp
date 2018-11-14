/**
 * @file
 */
#include "bi/common/Overloaded.hpp"

#include "bi/exception/all.hpp"

template<class ObjectType>
bi::Overloaded<ObjectType>::Overloaded(ObjectType* o) {
  add(o);
}

template<class ObjectType>
bool bi::Overloaded<ObjectType>::contains(ObjectType* o) {
  return overloads.contains(o);
}

template<class ObjectType>
ObjectType* bi::Overloaded<ObjectType>::get(ObjectType* o) {
  return overloads.get(o);
}

template<class ObjectType>
void bi::Overloaded<ObjectType>::add(ObjectType* o) {
  /* pre-condition */
  assert(!contains(o));

  overloads.insert(o);
}

template<class ObjectType>
int bi::Overloaded<ObjectType>::size() const {
  return overloads.size();
}

template<class ObjectType>
ObjectType* bi::Overloaded<ObjectType>::front() const {
  assert(overloads.size() == 1);
  return *overloads.begin();
}

template<class ObjectType>
void bi::Overloaded<ObjectType>::set(const Annotation a) {
  for (auto o : overloads) {
    o->set(a);
  }
}

template class bi::Overloaded<bi::Unknown>;
template class bi::Overloaded<bi::Function>;
template class bi::Overloaded<bi::Fiber>;
template class bi::Overloaded<bi::MemberFunction>;
template class bi::Overloaded<bi::MemberFiber>;
template class bi::Overloaded<bi::BinaryOperator>;
template class bi::Overloaded<bi::UnaryOperator>;
