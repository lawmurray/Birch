/**
 * @file
 */
#include "bi/common/Overloaded.hpp"

#include "bi/expression/Call.hpp"
#include "bi/expression/BinaryCall.hpp"
#include "bi/expression/UnaryCall.hpp"
#include "bi/statement/Function.hpp"
#include "bi/statement/Coroutine.hpp"
#include "bi/statement/MemberFunction.hpp"
#include "bi/statement/MemberCoroutine.hpp"
#include "bi/statement/BinaryOperator.hpp"
#include "bi/statement/UnaryOperator.hpp"

template<class ObjectType>
bi::Overloaded<ObjectType>::Overloaded(ObjectType* o) {
  add(o);
}

template<class ObjectType>
bool bi::Overloaded<ObjectType>::contains(ObjectType* o) const {
  return overloads.contains(o);
}

template<class ObjectType>
void bi::Overloaded<ObjectType>::add(ObjectType* o) {
  /* pre-condition */
  assert(!contains(o));

  overloads.insert(o);
}

template<class ObjectType>
void bi::Overloaded<ObjectType>::resolve(OverloadedCall<ObjectType>* o) {
  overloads.match(o, o->matches);
}

template class bi::Overloaded<bi::Function>;
template class bi::Overloaded<bi::Coroutine>;
template class bi::Overloaded<bi::MemberFunction>;
template class bi::Overloaded<bi::MemberCoroutine>;
template class bi::Overloaded<bi::BinaryOperator>;
template class bi::Overloaded<bi::UnaryOperator>;
