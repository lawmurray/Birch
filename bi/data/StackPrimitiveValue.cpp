/**
 * @file
 */
#include "bi/data/StackPrimitiveValue.hpp"

template<class Type>
bi::PrimitiveValue<Type,bi::StackGroup>::PrimitiveValue(
    const EmptyFrame& frame, const char* name, const StackGroup& group) {
  //
}

template<class Type>
bi::PrimitiveValue<Type,bi::StackGroup>::PrimitiveValue(const Type& value,
    const EmptyFrame& frame, const char* name, const StackGroup& group) :
    PrimitiveValue(frame, name, group) {
  copy(*this, value);
}

template<class Type>
bi::PrimitiveValue<Type,bi::StackGroup>::PrimitiveValue(
    const PrimitiveValue<Type,HeapGroup>& value, const EmptyFrame& frame,
    const char* name, const StackGroup& group) :
    PrimitiveValue(frame, name, group) {
  copy(*this, value);
}

template<class Type>
bi::PrimitiveValue<Type,bi::StackGroup>& bi::PrimitiveValue<Type,
    bi::StackGroup>::operator=(const PrimitiveValue<Type,StackGroup>& o) {
  copy(*this, o);
  return *this;
}

template<class Type>
bi::PrimitiveValue<Type,bi::StackGroup>& bi::PrimitiveValue<Type,
    bi::StackGroup>::operator=(const Type& o) {
  copy(*this, o);
  return *this;
}

template<class Type>
bi::PrimitiveValue<Type,bi::StackGroup>::operator Type&() {
  return value;
}

template<class Type>
bi::PrimitiveValue<Type,bi::StackGroup>::operator const Type&() const {
  return value;
}

template class bi::PrimitiveValue<unsigned char,bi::StackGroup>;
template class bi::PrimitiveValue<int64_t,bi::StackGroup>;
template class bi::PrimitiveValue<int32_t,bi::StackGroup>;
template class bi::PrimitiveValue<float,bi::StackGroup>;
template class bi::PrimitiveValue<double,bi::StackGroup>;
template class bi::PrimitiveValue<const char*,bi::StackGroup>;
template class bi::PrimitiveValue<std::function<void()>,bi::StackGroup>;
