/**
 * @file
 */
#include "bi/data/StackPrimitiveValue.hpp"

template<class Type>
bi::PrimitiveValue<Type,bi::StackGroup>::PrimitiveValue(
    const EmptyFrame& frame, const char* name, const StackGroup& group) :
    group(group) {
  this->group.create(*this, frame, name);
}

template<class Type>
bi::PrimitiveValue<Type,bi::StackGroup>::PrimitiveValue(
    const PrimitiveValue<Type,HeapGroup>& o) :
    value(*o.ptr) {
  //
}

template<class Type>
bi::PrimitiveValue<Type,bi::StackGroup>::PrimitiveValue(
    const PrimitiveValue<Type,NetCDFGroup>& o) {
  *this = o;
}

template<class Type>
bi::PrimitiveValue<Type,bi::StackGroup>::PrimitiveValue(const Type& value,
    const StackGroup& group) :
    value(value),
    group(group) {
  this->group.create(*this);
}

template<class Type>
bi::PrimitiveValue<Type,bi::StackGroup>::PrimitiveValue(
    const PrimitiveValue<Type,StackGroup>& o, const EmptyFrame& frame,
    const EmptyView& view) :
    value(o.value),
    group(o.group) {
}

//template<class Type>
//bi::PrimitiveValue<Type,bi::StackGroup>::PrimitiveValue(
//    const PrimitiveValue<Type,StackGroup>& o) :
//    value(o.value),
//    group(o.group) {
//  //
//}
//
//template<class Type>
//bi::PrimitiveValue<Type,bi::StackGroup>::PrimitiveValue(
//    PrimitiveValue<Type,StackGroup> && o) :
//    value(o.value),
//    group(o.group) {
//  //
//}

template<class Type>
bi::PrimitiveValue<Type,bi::StackGroup>::~PrimitiveValue() {
  group.release(*this);
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
