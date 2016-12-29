/**
 * @file
 */
#include "bi/data/NetCDFPrimitiveValue.hpp"

template<class Type>
bi::PrimitiveValue<Type,bi::NetCDFGroup>::~PrimitiveValue() {
  group.release(*this);
}

template<class Type>
bi::PrimitiveValue<Type,bi::NetCDFGroup>& bi::PrimitiveValue<Type,
    bi::NetCDFGroup>::operator=(const PrimitiveValue<Type,NetCDFGroup>& o) {
  copy(*this, o);
  return *this;
}

template<class Type>
bi::PrimitiveValue<Type,bi::NetCDFGroup>& bi::PrimitiveValue<Type,
    bi::NetCDFGroup>::operator=(const Type& o) {
  copy(*this, o);
  return *this;
}

template<class Type>
bi::PrimitiveValue<Type,bi::NetCDFGroup>::operator Type() const {
  Type value;
  get(group.ncid, varid, convolved.offsets.data(), &value);
  return value;
}

template class bi::PrimitiveValue<unsigned char,bi::NetCDFGroup>;
template class bi::PrimitiveValue<int64_t,bi::NetCDFGroup>;
template class bi::PrimitiveValue<int32_t,bi::NetCDFGroup>;
template class bi::PrimitiveValue<float,bi::NetCDFGroup>;
template class bi::PrimitiveValue<double,bi::NetCDFGroup>;
