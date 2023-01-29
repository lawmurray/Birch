/**
 * @file
 */
#pragma once

#include "numbirch/array/ArrayControl.hpp"
#include "numbirch/array/Array.hpp"

namespace numbirch {
/**
 * @internal
 * 
 * Temporary type that injects coordination operations into the device stream
 * when slicing a buffer.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
 *
 * Sliced is not intended to be used explicitly. It is the parameter type of
 * sliced(), triggering an implicit conversion from Array to Sliced to raw
 * pointer return. The constructor (or raw pointer conversion) and destructor
 * provide injection opportunities either side of a task being enqueued in the
 * device stream. The usage idiom is:
 * 
 * y = f(sliced(x), stride(x));
 * 
 * where `x` is of Array type; `x` is implicitly converted to Sliced type as
 * the argument to sliced(), enabling the first injection via its constructor,
 * then converted to raw pointer type as the return value of sliced(),
 * enabling a second injection via the conversion operator, then `f` is called
 * to enqueue the kernel, then the Sliced object is destroyed, enabling a
 * third injection via its destructor.
 * 
 * @see Diced
 */
template<class T>
class Sliced {
public:
  static_assert(!std::is_const_v<T>, "Sliced cannot have const value type");

  /**
   * Constructor.
   */
  Sliced(ArrayControl* ctl, const int64_t offset, const bool write) :
      ctl(ctl),
      offset(offset),
      write(write) {
    //
  }

  /**
   * Constructor.
   */
  template<int D>
  Sliced(Array<T,D>& x) :
      ctl(x.control()),
      offset(x.offset()),
      write(true) {
    //
  }

  /**
   * Constructor.
   */
  template<int D>
  Sliced(const Array<T,D>& x) :
      ctl(x.control()),
      offset(x.offset()),
      write(false) {
    //
  }

  /**
   * Destructor.
   */
  ~Sliced() {
    if (ctl) {
      if (write) {
        event_record_write(ctl->writeEvt);
      } else {
        event_record_read(ctl->readEvt);
      }
    }
  }

  /**
   * Get raw pointer to buffer.
   */
  T* data() const {
    if (ctl) {
      event_join(ctl->writeEvt);
      event_join(ctl->readEvt);
      return static_cast<T*>(ctl->buf) + offset;
    } else {
      return nullptr;
    }
  }

  /**
   * Get raw pointer to buffer.
   */
  operator T*() const {
    return data();
  }

private:
  /**
   * Buffer control block.
   */
  ArrayControl* ctl;

  /**
   * Offset into buffer.
   */
  int64_t offset;

  /**
   * Is this for a write?
   */
  bool write;
};

template<class T, int D>
Sliced(const Array<T,D>& x) -> Sliced<T>;

}
