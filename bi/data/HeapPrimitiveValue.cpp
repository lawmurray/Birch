/**
 * @file
 */
#include "bi/data/HeapPrimitiveValue.hpp"

template<class Type>
bi::PrimitiveValue<Type,bi::HeapGroup>::PrimitiveValue(
    PrimitiveValue<Type,StackGroup>& o, const HeapGroup& group) :
    ptr(&o.value),
    group(group),
    own(false) {
  //
}

template<class Type>
bi::PrimitiveValue<Type,bi::HeapGroup>::PrimitiveValue(
    const PrimitiveValue<Type,HeapGroup>& o) :
    ptr(o.ptr),
    group(o.group),
    own(false) {
  //
}

template<class Type>
bi::PrimitiveValue<Type,bi::HeapGroup>::PrimitiveValue(
    PrimitiveValue<Type,HeapGroup> && o) :
    ptr(o.ptr),
    group(o.group),
    own(o.own) {
  o.own = false;  // ownership moves
}

template<class Type>
bi::PrimitiveValue<Type,bi::HeapGroup>::~PrimitiveValue() {
  if (own) {
    group.release(*this);
  }
}

template<class Type>
bi::PrimitiveValue<Type,bi::HeapGroup>& bi::PrimitiveValue<Type,bi::HeapGroup>::operator=(
    const PrimitiveValue<Type,HeapGroup>& o) {
  copy(*this, o);
  return *this;
}

template<class Type>
bi::PrimitiveValue<Type,bi::HeapGroup>& bi::PrimitiveValue<Type,bi::HeapGroup>::operator=(
    const Type& o) {
  copy(*this, o);
  return *this;
}

template<class Type>
bi::PrimitiveValue<Type,bi::HeapGroup>::operator Type&() {
  return *ptr;
}

template<class Type>
bi::PrimitiveValue<Type,bi::HeapGroup>::operator const Type&() const {
  return *ptr;
}

template class bi::PrimitiveValue<unsigned char,bi::HeapGroup>;
template class bi::PrimitiveValue<int64_t,bi::HeapGroup>;
template class bi::PrimitiveValue<int32_t,bi::HeapGroup>;
template class bi::PrimitiveValue<float,bi::HeapGroup>;
template class bi::PrimitiveValue<double,bi::HeapGroup>;
template class bi::PrimitiveValue<std::function<void()>,bi::HeapGroup>;
