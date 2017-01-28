/**
 * @file
 *
 * Assignment functions between data.
 */
#pragma once

#include "bi/data/MemoryGroup.hpp"
#include "bi/data/NetCDFGroup.hpp"
#include "bi/data/Array.hpp"
#include "bi/data/netcdf.hpp"

namespace bi {
/**
 * Static computation of the greatest common divisor of two positive
 * integers. If either is zero, returns zero.
 */
template<int_t a, int_t b>
struct gcd {
  static const int_t value = gcd<b,a % b>::value;
};
template<int_t a>
struct gcd<a,a> {
  static const int_t value = a;
};
template<int_t a>
struct gcd<a,0> {
  static const int_t value = a;
};
template<int_t b>
struct gcd<0,b> {
  static const int_t value = b;
};
template<>
struct gcd<0,0> {
  static const int_t value = 0;
};

/**
 * Greatest common divisor of two positive integers.
 */
int_t common_length(const int_t a, const int_t b);

/**
 * Range with length that is the greatest common divisor of the lengths of
 * two other ranges.
 */
template<int_t offset_value1, int_t length_value1, int_t stride_value1,
    int_t offset_value2, int_t length_value2, int_t stride_value2>
auto common_range(const Range<offset_value1,length_value1,stride_value1>& o1,
    const Range<offset_value2,length_value2,stride_value2>& o2) {
  /* pre-condition */
  assert(o1.stride == 1);
  assert(o2.stride == 1);

  static const int_t offset_value = 0;
  static const int_t length_value = gcd<length_value1,length_value2>::value;
  static const int_t stride_value = 1;

  const int_t offset = 0;
  const int_t length = common_length(o1.length, o2.length);
  const int_t stride = 1;

  return Range<offset_value,length_value,stride_value>(offset, length, stride);
}

/**
 * Frame with span lengths that are the greatest common divisor of the
 * lengths of the spans of two other frames.
 */
template<class View1, class View2>
auto common_view(const View1& o1, const View2& o2) {
  auto tail = common_view(o1.tail, o2.tail);
  auto head = common_range(o1.head, o2.head);

  return NonemptyView<decltype(tail),decltype(head)>(tail, head);
}
inline EmptyView common_view(const EmptyView& o1, const EmptyView& o2) {
  return EmptyView();
}

/**
 * Catch-all for unimplemented copies.
 */
template<class Type1, class Type2>
void copy(Type1& o1, const Type2& o2) {
  assert(false);
}

/**
 * @name Scalar copies from primitive values
 */
//@{
template<class Type>
void copy(PrimitiveValue<Type,MemoryGroup>& dst, const Type& src) {
  *dst.ptr = src;
}

template<class Type>
void copy(PrimitiveValue<Type,NetCDFGroup>& dst, const Type& src) {
  put(dst.group.ncid, dst.varid, dst.convolved.offsets.data(), &src);
}
//@}

/**
 * @name Scalar copies
 */
//@{
template<class Type>
void copy(PrimitiveValue<Type,MemoryGroup>& dst,
    const PrimitiveValue<Type,MemoryGroup>& src) {
  *dst.ptr = *src.ptr;
}

template<class Type>
void copy(PrimitiveValue<Type,MemoryGroup>& dst,
    const PrimitiveValue<Type,NetCDFGroup>& src) {
  get(src.group.ncid, src.varid, src.convolved.offsets.data(), dst.ptr);
}

template<class Type>
void copy(PrimitiveValue<Type,NetCDFGroup>& dst,
    const PrimitiveValue<Type,MemoryGroup>& src) {
  put(dst.group.ncid, dst.varid, dst.convolved.offsets.data(), src.ptr);
}

template<class Type>
void copy(PrimitiveValue<Type,NetCDFGroup>& dst,
    const PrimitiveValue<Type,NetCDFGroup>& src) {
  Type value;
  get(src.group.ncid, src.varid, src.convolved.offsets.data(), &value);
  put(dst.group.ncid, dst.varid, dst.convolved.offsets.data(), &value);
}
//@}

/**
 * @name Array copies
 */
//@{
/**
 * Copy for all arrays, iterating over contiguous chunks.
 */
template<class Type, class Group1, class Frame1, class Group2, class Frame2>
void copy(Array<PrimitiveValue<Type,Group1>,Frame1>& dst,
    const Array<PrimitiveValue<Type,Group2>,Frame2>& src) {
  /* pre-condition */
  assert(dst.frame.conforms(src.frame));

  auto block = common_view(dst.frame.block(), src.frame.block());
  auto iter1 = dst.begin(block);
  auto iter2 = src.begin(block);
  auto end1 = dst.end(block);
  auto end2 = src.end(block);

  for (; iter1 != end1; ++iter1, ++iter2) {
    /* pre-conditions */
    auto a1 = *iter1;
    auto a2 = *iter2;

    assert(a1.frame.contiguous());
    assert(a2.frame.contiguous());
    assert(a1.frame.length == a2.frame.length);

    contiguous_copy(a1, a2);
  }
  assert(iter2 == end2);
}

/*
 * Contiguous copies for arrays between groups.
 */
template<class Type, class Frame1, class Frame2>
void contiguous_copy(Array<PrimitiveValue<Type,MemoryGroup>,Frame1>& dst,
    const Array<PrimitiveValue<Type,MemoryGroup>,Frame2>& src) {
  memcpy(dst.value.ptr, src.value.ptr, dst.frame.lead * sizeof(Type));
}

template<class Type, class Frame1, class Frame2>
void contiguous_copy(Array<PrimitiveValue<Type,MemoryGroup>,Frame1>& dst,
    const Array<PrimitiveValue<Type,NetCDFGroup>,Frame2>& src) {
  get(src.value.group.ncid, src.value.varid,
      src.value.convolved.offsets.data(), src.value.convolved.lengths.data(),
      dst.value.ptr);
}

template<class Type, class Frame1, class Frame2>
void contiguous_copy(Array<PrimitiveValue<Type,NetCDFGroup>,Frame1>& dst,
    const Array<PrimitiveValue<Type,MemoryGroup>,Frame2>& src) {
  put(dst.value.group.ncid, dst.value.varid,
      dst.value.convolved.offsets.data(), dst.value.convolved.lengths.data(),
      src.value.ptr);
}

template<class Type, class Frame1, class Frame2>
void contiguous_copy(Array<PrimitiveValue<Type,NetCDFGroup>,Frame1>& dst,
    const Array<PrimitiveValue<Type,NetCDFGroup>,Frame2>& src) {
  Type tmp[src.frame.length];
  get(src.value.group.ncid, src.value.varid,
      src.value.convolved.offsets.data(), src.value.convolved.lengths.data(),
      tmp);
  put(dst.value.group.ncid, dst.value.varid,
      dst.value.convolved.offsets.data(), dst.value.convolved.lengths.data(),
      tmp);
}
//@}

}
