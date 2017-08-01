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
#include "bi/exception/all.hpp"

template<class ObjectType>
bi::Overloaded<ObjectType>::Overloaded(ObjectType* o) {
  add(o);
}

template<class ObjectType>
bool bi::Overloaded<ObjectType>::contains(ObjectType* o) const {
  return std::find(overloads.begin(), overloads.end(), o) != overloads.end();
}

template<class ObjectType>
void bi::Overloaded<ObjectType>::add(ObjectType* o) {
  /* pre-condition */
  assert(!contains(o));

  overloads.push_back(o);
  overloadTypes.insert(o->type);
}

template class bi::Overloaded<bi::Function>;
template class bi::Overloaded<bi::Coroutine>;
template class bi::Overloaded<bi::MemberFunction>;
template class bi::Overloaded<bi::MemberCoroutine>;
template class bi::Overloaded<bi::BinaryOperator>;
template class bi::Overloaded<bi::UnaryOperator>;
