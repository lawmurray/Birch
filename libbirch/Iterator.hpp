/**
 * @file
 */
#pragma once

#include "libbirch/Frame.hpp"

namespace libbirch {
#pragma omp declare target
/**
 * Iterator over an array.
 *
 * @ingroup libbirch
 *
 * @tparam T Value type.
 * @tparam F Frame type.
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
  Iterator(T* ptr, const F& frame, int64_t serial = 0) :
      frame(frame),
      ptr(ptr),
      serial(serial) {
    //
  }

  Iterator(const Iterator& o) = default;

  T* get() const {
    return ptr + frame.offset(serial);
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
   * Frame.
   */
  F frame;

  /**
   * Buffer.
   */
  T* ptr;

  /**
   * Serialised offset into the frame.
   */
  int64_t serial;
};
#pragma omp end declare target

/**
 * Is @p iter inside the range @p begin to @p end?
 */
template<class T, class F, class G>
bool inside(const Iterator<T,F>& begin, const Iterator<T,F>& end,
     const Iterator<T,G>& iter) {
  return begin <= iter && iter <= end;
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
