/**
 * @file
 */
#pragma once

#include "bi/lib/Frame.hpp"
#include "bi/lib/reloc_ptr.hpp"

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
  Iterator(const reloc_ptr<Type>& ptr, const Frame& frame) :
      frame(frame),
      ptr(ptr),
      serial(0) {
    //
  }

  Type& operator*() {
    return *(ptr + frame.offset(serial));
  }

  const Type& operator*() const {
    return *(ptr + frame.offset(serial));
  }

  bool operator==(const Iterator<Type,Frame>& o) const {
    return ptr == o.ptr && serial == o.serial;
  }

  bool operator!=(const Iterator<Type,Frame>& o) const {
    return !(*this == o);
  }

  Iterator<Type,Frame>& operator+=(const ptrdiff_t i) {
    serial += i;
    return *this;
  }

  Iterator<Type,Frame> operator+(const ptrdiff_t i) const {
    Iterator<Type,Frame> result(*this);
    result += i;
    return result;
  }

  Iterator<Type,Frame>& operator-=(const ptrdiff_t i) {
    serial -= i;
    return *this;
  }

  Iterator<Type,Frame> operator-(const ptrdiff_t i) const {
    Iterator<Type,Frame> result(*this);
    result -= i;
    return result;
  }

  Iterator<Type,Frame>& operator++() {
    serial += 1;
    return *this;
  }

  Iterator<Type,Frame> operator++(int) {
    Iterator<Type,Frame> result(*this);
    ++*this;
    return result;
  }

  Iterator<Type,Frame>& operator--() {
    serial -= 1;
    return *this;
  }

  Iterator<Type,Frame> operator--(int) {
    Iterator<Type,Frame> result(*this);
    --*this;
    return result;
  }

//protected:
  /**
   * Frame.
   */
  Frame frame;

  /**
   * Value.
   */
  reloc_ptr<Type> ptr;

  /**
   * Serialised offset into the frame.
   */
  ptrdiff_t serial;
};
}
