/**
 * @file
 */
#include "bi/data/MemoryPrimitiveValue.hpp"

template<class Type>
bi::PrimitiveValue<Type,bi::MemoryGroup>::PrimitiveValue(
    const EmptyFrame& frame, const char* name, const MemoryGroup& group) :
    group(group),
    own(true) {
  this->group.create(*this, frame, name);
}

template<class Type>
bi::PrimitiveValue<Type,bi::MemoryGroup>::PrimitiveValue(
    const PrimitiveValue<Type,MemoryGroup>& o) :
    group(o.group),
    own(true) {
  this->group.create(*this);
  *this = o;
}

template<class Type>
bi::PrimitiveValue<Type,bi::MemoryGroup>::PrimitiveValue(const Type& value) :
    own(true) {
  this->group.create(*this, value);
}

template<class Type>
bi::PrimitiveValue<Type,bi::MemoryGroup>::PrimitiveValue(
    PrimitiveValue<Type,MemoryGroup> && o) :
    ptr(o.ptr),
    group(o.group),
    own(o.own) {
  o.own = false;  // ownership moves
}

template<class Type>
bi::PrimitiveValue<Type,bi::MemoryGroup>::~PrimitiveValue() {
  if (own) {
    group.release(*this);
  }
}

template<class Type>
bi::PrimitiveValue<Type,bi::MemoryGroup>& bi::PrimitiveValue<Type,
    bi::MemoryGroup>::operator=(const PrimitiveValue<Type,MemoryGroup>& o) {
  copy(*this, o);
  return *this;
}

template<class Type>
bi::PrimitiveValue<Type,bi::MemoryGroup>& bi::PrimitiveValue<Type,
    bi::MemoryGroup>::operator=(PrimitiveValue<Type,MemoryGroup> && o) {
  std::swap(ptr, o.ptr);
  std::swap(group, o.group);
  std::swap(own, o.own);
  return *this;
}

template<class Type>
bi::PrimitiveValue<Type,bi::MemoryGroup>& bi::PrimitiveValue<Type,
    bi::MemoryGroup>::operator=(const Type& o) {
  copy(*this, o);
  return *this;
}

template<class Type>
bi::PrimitiveValue<Type,bi::MemoryGroup>::operator Type&() {
  return *ptr;
}

template<class Type>
bi::PrimitiveValue<Type,bi::MemoryGroup>::operator const Type&() const {
  return *ptr;
}

template class bi::PrimitiveValue<unsigned char,bi::MemoryGroup>;
template class bi::PrimitiveValue<int64_t,bi::MemoryGroup>;
template class bi::PrimitiveValue<int32_t,bi::MemoryGroup>;
template class bi::PrimitiveValue<float,bi::MemoryGroup>;
template class bi::PrimitiveValue<double,bi::MemoryGroup>;
template class bi::PrimitiveValue<std::function<void()>,bi::MemoryGroup>;
