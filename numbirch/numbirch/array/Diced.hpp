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
 * Temporary type that injects coordination operations for the host when
 * dicing a buffer.
 * 
 * @ingroup array
 * 
 * @tparam T Value type.
 *
 * Diced is not intended to be used explicitly. It is the parameter type of
 * diced(), triggering an implicit conversion from Array to Diced to raw
 * pointer return. The constructor (or raw pointer conversion) and destructor
 * provide injection opportunities either side of a task. Specifically, Diced
 * is used to inject a device stream synchronization before the task.
 * 
 * @see Sliced
 */
template<class T>
class Diced {
public:
  static_assert(!std::is_const_v<T>, "Diced cannot have const value type");

  /**
   * Constructor.
   */
  Diced(ArrayControl* ctl, const int64_t offset) :
      ctl(ctl),
      offset(offset) {
    //
  }

  /**
   * Constructor.
   */
  template<int D>
  Diced(Array<T,D>& x) :
      ctl(x.control()),
      offset(x.offset()) {
    //
  }

  /**
   * Constructor.
   */
  template<int D>
  Diced(const Array<T,D>& x) :
      ctl(x.control()),
      offset(x.offset()) {
    //
  }

  /**
   * Get raw pointer to buffer.
   */
  T* data() const {
    if (ctl) {
      array_wait(ctl);
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
};

template<class T, int D>
Diced(const Array<T,D>& x) -> Diced<T>;

}
