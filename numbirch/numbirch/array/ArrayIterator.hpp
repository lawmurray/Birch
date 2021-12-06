/**
 * @file
 */
#pragma once

#include "numbirch/array/external.hpp"

namespace numbirch {
/**
 * Iterator over Array.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
 * @tparam D Number of dimensions.
 */
template<class T, int D>
class ArrayIterator {
public:
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using reference = T&;
  using iterator_category = std::random_access_iterator_tag;

  explicit ArrayIterator(T* buf, const ArrayShape<D> shp,
      const difference_type pos) :
      buf(buf),
      shp(shp),
      pos(pos) {
    //
  }

  reference operator[](const difference_type i) {
    return buf[shp.offset(pos + i)];
  }

  difference_type operator-(const ArrayIterator& o) const {
    return pos - o.pos;
  }

  reference operator*() {
    return *get();
  }

  reference operator*() const {
    return *get();
  }

  pointer operator->() {
    return get();
  }

  pointer operator->() const {
    return get();
  }

  bool operator==(const ArrayIterator& o) const {
    return get() == o.get();
  }

  bool operator!=(const ArrayIterator& o) const {
    return get() != o.get();
  }

  bool operator<=(const ArrayIterator& o) const {
    return get() <= o.get();
  }

  bool operator<(const ArrayIterator& o) const {
    return get() < o.get();
  }

  bool operator>=(const ArrayIterator& o) const {
    return get() >= o.get();
  }

  bool operator>(const ArrayIterator& o) const {
    return get() > o.get();
  }

  ArrayIterator& operator+=(const difference_type i) {
    pos += i;
    return *this;
  }

  ArrayIterator operator+(const difference_type i) const {
    ArrayIterator result(*this);
    result += i;
    return result;
  }

  ArrayIterator& operator-=(const difference_type i) {
    pos -= i;
    return *this;
  }

  ArrayIterator operator-(const difference_type i) const {
    ArrayIterator result(*this);
    result -= i;
    return result;
  }

  ArrayIterator& operator++() {
    ++pos;
    return *this;
  }

  ArrayIterator operator++(int) {
    ArrayIterator result(*this);
    ++pos;
    return result;
  }

  ArrayIterator& operator--() {
    --pos;
    return *this;
  }

  ArrayIterator operator--(int) {
    ArrayIterator result(*this);
    --pos;
    return result;
  }

protected:
  /**
   * Raw pointer for the current position.
   */
  pointer get() const {
    return buf + shp.offset(pos);
  }

  /**
   * Buffer.
   */
  T* buf;

  /**
   * Shape.
   */
  ArrayShape<D> shp;

  /**
   * Position.
   */
  difference_type pos;
};
}
