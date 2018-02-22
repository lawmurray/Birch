/**
 * @file
 */
#pragma once

#include "libbirch/Frame.hpp"

namespace bi {
/**
 * Iterator over an array.
 *
 * @ingroup libbirch
 *
 * @tparam T Value type.
 * @tparam F F type.
 */
template<class T, class F = EmptyFrame>
class Iterator : public std::iterator<std::bidirectional_iterator_tag,T> {
public:
  /**
   * Constructor.
   *
   * @param ptr Buffer.
   * @param frame F.
   */
  Iterator(T* ptr, const F& frame) :
      frame(frame),
      ptr(ptr),
      serial(0) {
    //
  }

  T& operator*() {
    return *(ptr + frame.offset(serial));
  }

  const T& operator*() const {
    return *(ptr + frame.offset(serial));
  }

  T* operator->() {
    return ptr + frame.offset(serial);
  }

  T* const operator->() const {
    return ptr + frame.offset(serial);
  }

  bool operator==(const Iterator<T,F>& o) const {
    return ptr == o.ptr && serial == o.serial;
  }

  bool operator!=(const Iterator<T,F>& o) const {
    return !(*this == o);
  }

  Iterator<T,F>& operator+=(const ptrdiff_t i) {
    serial += i;
    return *this;
  }

  Iterator<T,F> operator+(const ptrdiff_t i) const {
    Iterator<T,F> result(*this);
    result += i;
    return result;
  }

  Iterator<T,F>& operator-=(const ptrdiff_t i) {
    serial -= i;
    return *this;
  }

  Iterator<T,F> operator-(const ptrdiff_t i) const {
    Iterator<T,F> result(*this);
    result -= i;
    return result;
  }

  Iterator<T,F>& operator++() {
    serial += 1;
    return *this;
  }

  Iterator<T,F> operator++(int) {
    Iterator<T,F> result(*this);
    ++*this;
    return result;
  }

  Iterator<T,F>& operator--() {
    serial -= 1;
    return *this;
  }

  Iterator<T,F> operator--(int) {
    Iterator<T,F> result(*this);
    --*this;
    return result;
  }

protected:
  /**
   * F.
   */
  F frame;

  /**
   * Value.
   */
  T* ptr;

  /**
   * Serialised offset into the frame.
   */
  ptrdiff_t serial;
};
}
