/**
 * @file
 */
#include "bi/data/RefPrimitiveValue.hpp"

template<class Type>
bi::PrimitiveValue<Type,bi::RefGroup>::PrimitiveValue(const Type& o,
    const EmptyFrame& frame, const char* name, const RefGroup& group) :
    value(const_cast<Type&>(o)) {
  //
}

template<class Type>
bi::PrimitiveValue<Type,bi::RefGroup>::PrimitiveValue(
    const PrimitiveValue<Type,HeapGroup>& o, const EmptyFrame& frame,
    const char* name, const RefGroup& group) :
    value(const_cast<Type&>(*o.ptr)) {
  //
}

template<class Type>
bi::PrimitiveValue<Type,bi::RefGroup>::PrimitiveValue(
    const PrimitiveValue<Type,StackGroup>& o, const EmptyFrame& frame,
    const char* name, const RefGroup& group) :
    value(const_cast<Type&>(o.value)) {
  //
}

template<class Type>
bi::PrimitiveValue<Type,bi::RefGroup>& bi::PrimitiveValue<Type,bi::RefGroup>::operator=(
    const PrimitiveValue<Type,RefGroup>& o) {
  copy(*this, o);
  return *this;
}

template<class Type>
bi::PrimitiveValue<Type,bi::RefGroup>& bi::PrimitiveValue<Type,bi::RefGroup>::operator=(
    const Type& o) {
  copy(*this, o);
  return *this;
}

template<class Type>
bi::PrimitiveValue<Type,bi::RefGroup>::operator Type&() {
  return value;
}

template<class Type>
bi::PrimitiveValue<Type,bi::RefGroup>::operator const Type&() const {
  return value;
}

template class bi::PrimitiveValue<unsigned char,bi::RefGroup>;
template class bi::PrimitiveValue<int64_t,bi::RefGroup>;
template class bi::PrimitiveValue<int32_t,bi::RefGroup>;
template class bi::PrimitiveValue<float,bi::RefGroup>;
template class bi::PrimitiveValue<double,bi::RefGroup>;
template class bi::PrimitiveValue<const char*,bi::RefGroup>;
template class bi::PrimitiveValue<std::function<void()>,bi::RefGroup>;
