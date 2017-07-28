/**
 * @file
 */
#include "bi/common/Overloaded.hpp"

#include "bi/statement/Function.hpp"
#include "bi/statement/Coroutine.hpp"
#include "bi/statement/MemberFunction.hpp"
#include "bi/statement/MemberCoroutine.hpp"
#include "bi/statement/BinaryOperator.hpp"
#include "bi/statement/UnaryOperator.hpp"
#include "bi/statement/AssignmentOperator.hpp"

template<class ObjectType>
bi::Overloaded<ObjectType>::Overloaded(ObjectType* o) :
    Named(o->name) {
  add(o);
}

template<class ObjectType>
bool bi::Overloaded<ObjectType>::contains(ObjectType* o) const {
  return objects.contains(o);
}

template<class ObjectType>
void bi::Overloaded<ObjectType>::add(ObjectType* o) {
  /* pre-condition */
  assert(!contains(o));

  objects.insert(o);
}

template<class ObjectType>
void bi::Overloaded<ObjectType>::import(Overloaded<ObjectType>& o) {
  for (auto object : o.objects) {
    if (!contains(object)) {
      add(object);
    }
  }
}

/*
 * Explicit instantiations.
 */
template class bi::Overloaded<bi::Function>;
template class bi::Overloaded<bi::Coroutine>;
template class bi::Overloaded<bi::MemberFunction>;
template class bi::Overloaded<bi::MemberCoroutine>;
template class bi::Overloaded<bi::BinaryOperator>;
template class bi::Overloaded<bi::UnaryOperator>;
template class bi::Overloaded<bi::AssignmentOperator>;
