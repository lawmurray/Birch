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
bi::PrimitiveValue<Type,bi::MemoryGroup>::PrimitiveValue(const Type& value) :
    own(true) {
  this->group.create(*this);
  this->group.fill(*this, value);
}

template<class Type>
bi::PrimitiveValue<Type,bi::MemoryGroup>::PrimitiveValue(
    const PrimitiveValue<Type,MemoryGroup>& o) :
    own(true) {
  this->group.create(*this);
  *this = o;
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
  if (ptr == o.ptr && o.own) {
    /* just take ownership */
    assert(!own);  // there should only be one owner
    own = true;
    o.own = false;
  } else {
    /* copy assignment */
    own = true;
    this->group.create(*this);
    *this = o;
  }
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

#include "bi/method/RandomState.hpp"

template class bi::PrimitiveValue<unsigned char,bi::MemoryGroup>;
template class bi::PrimitiveValue<int64_t,bi::MemoryGroup>;
template class bi::PrimitiveValue<int32_t,bi::MemoryGroup>;
template class bi::PrimitiveValue<float,bi::MemoryGroup>;
template class bi::PrimitiveValue<double,bi::MemoryGroup>;
template class bi::PrimitiveValue<bi::RandomState,bi::MemoryGroup>;
