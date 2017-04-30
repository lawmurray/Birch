/**
 * @file
 */
#pragma once

#include "bi/data/Frame.hpp"

namespace bi {
/**
 * Iterator over an array.
 *
 * @ingroup library
 *
 * @tparam Type Value type.
 * @tparam Frame Frame type.
 */
template<class Type, class Frame = EmptyFrame>
class Iterator {
public:
  /**
   * Constructor.
   *
   * @param ptr Value.
   * @param frame Frame.
   */
  Iterator(Type* ptr, const Frame& frame) :
      frame(frame),
      ptr(ptr),
      serial(0) {
    //
  }

  Type& operator*() {
    return ptr[frame.offset(serial)];
  }

  const Type& operator*() const {
    return ptr[frame.offset(serial)];
  }

  bool operator==(const Iterator<Type,Frame>& o) const {
    return ptr == o.ptr && serial == o.serial;
  }

  bool operator!=(const Iterator<Type,Frame>& o) const {
    return !(*this == o);
  }

  Iterator<Type,Frame>& operator+=(const int_t i) {
    serial += i;
    return *this;
  }

  Iterator<Type,Frame>& operator+(const int_t i) const {
    auto result = *this;
    result += i;
    return result;
  }

  Iterator<Type,Frame>& operator-=(const int_t i) {
    serial -= i;
    return *this;
  }

  Iterator<Type,Frame>& operator-(const int_t i) const {
    auto result = *this;
    result -= i;
    return result;
  }

  Iterator<Type,Frame>& operator++() {
    serial += 1;
    return *this;
  }

  Iterator<Type,Frame> operator++(int) {
    auto result = *this;
    ++*this;
    return result;
  }

  Iterator<Type,Frame>& operator--() {
    serial -= 1;
    return *this;
  }

  Iterator<Type,Frame> operator--(int) {
    auto result = *this;
    --*this;
    return result;
  }

protected:
  /**
   * Frame.
   */
  Frame frame;

  /**
   * Value.
   */
  Type* ptr;

  /**
   * Serialised offset into the frame.
   */
  int_t serial;
};
}
