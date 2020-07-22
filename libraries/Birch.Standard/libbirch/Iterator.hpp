/**
 * @file
 */
#pragma once

#include "libbirch/Shape.hpp"

namespace libbirch {
/**
 * Iterator over an array.
 *
 * @ingroup libbirch
 *
 * @tparam T Value type.
 * @tparam F Shape type.
 */
template<class T, class F = EmptyShape>
class Iterator : public std::iterator<std::bidirectional_iterator_tag,T> {
public:
  /**
   * Constructor.
   *
   * @param ptr Buffer.
   * @param shape F.
   */
  Iterator(T* ptr, const F& shape, int64_t serial = 0) :
      shape(shape),
      ptr(ptr),
      serial(serial) {
    //
  }

  Iterator(const Iterator& o) = default;

  T* get() const {
    return ptr + shape.offset(serial);
  }

  T& operator*() {
    return *get();
  }

  const T& operator*() const {
    return *get();
  }

  T* operator->() {
    return get();
  }

  T* operator->() const {
    return get();
  }

  bool operator==(const Iterator<T,F>& o) const {
    return get() == o.get();
  }

  bool operator!=(const Iterator<T,F>& o) const {
    return get() != o.get();
  }

  bool operator<=(const Iterator<T,F>& o) const {
    return get() <= o.get();
  }

  bool operator<(const Iterator<T,F>& o) const {
    return get() < o.get();
  }

  bool operator>=(const Iterator<T,F>& o) const {
    return get() >= o.get();
  }

  bool operator>(const Iterator<T,F>& o) const {
    return get() > o.get();
  }

  int64_t operator-(const Iterator<T,F>& o) const {
    return serial - o.serial;
  }

  Iterator<T,F>& operator+=(const int64_t i) {
    serial += i;
    return *this;
  }

  Iterator<T,F> operator+(const int64_t i) const {
    Iterator<T,F> result(*this);
    result += i;
    return result;
  }

  Iterator<T,F>& operator-=(const int64_t i) {
    serial -= i;
    return *this;
  }

  Iterator<T,F> operator-(const int64_t i) const {
    Iterator<T,F> result(*this);
    result -= i;
    return result;
  }

  Iterator<T,F>& operator++() {
    ++serial;
    return *this;
  }

  Iterator<T,F> operator++(int) {
    Iterator<T,F> result(*this);
    ++*this;
    return result;
  }

  Iterator<T,F>& operator--() {
    --serial;
    return *this;
  }

  Iterator<T,F> operator--(int) {
    Iterator<T,F> result(*this);
    --*this;
    return result;
  }

protected:
  /**
   * Shape.
   */
  F shape;

  /**
   * Buffer.
   */
  T* ptr;

  /**
   * Serialised offset into the shape.
   */
  int64_t serial;
};

/**
 * Is @p iter inside the range @p begin to @p end?
 */
template<class T, class F, class G>
bool inside(const Iterator<T,F>& begin, const Iterator<T,F>& end,
     const Iterator<T,G>& iter) {
  return begin <= iter && iter < end;
}

/**
 * Is @p iter inside the range @p begin to @p end?
 */
template<class T, class F, class U, class G>
bool inside(const Iterator<T,F>& begin, const Iterator<T,F>& end,
     const Iterator<U,G>& iter) {
  return false;
}
}
