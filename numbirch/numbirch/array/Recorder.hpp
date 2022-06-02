/**
 * @file
 */
#pragma once

#include "numbirch/memory.hpp"

namespace numbirch {
/**
 * Wrapper around a buffer that injects the recording of an event into the
 * device stream via its destructor.
 * 
 * @ingroup array
 * 
 * @tparam T Value type. If constant, then a read event is recorded by the
 * destructor, otherwise a write event.
 *
 * This is returned by numbirch::sliced(). The raw pointer can be accessed
 * with numbirch::data(), which simply calls the member function data() but
 * facilitates pass-through for non-Recorder objects. To obtain a raw pointer
 * to the underlying buffer of an Array object `x` then pass it to a function
 * `f`, the usage idiom is e.g.:
 * 
 * ```
 * y = f(data(sliced(x)), stride(x));
 * ```
 * 
 * The numbirch::sliced() call constructs the Recorder object, the
 * numbirch::data() call obtains the raw pointer from it, and after completion
 * of the statement, the destructor of the Recorder object records the event.
 */
template<class T>
class Recorder {
public:
  /**
   * Constructor.
   * 
   * @param buf Buffer.
   * @param evt Event to record.
   */
  Recorder(T* buf = nullptr, void* evt = 0) :
      buf(buf),
      evt(evt) {
    //
  }

  /**
   * Destructor.
   */
  ~Recorder() {
    if (buf && evt) {
      if constexpr (std::is_const_v<T>) {
        event_record_read(evt);
      } else {
        event_record_write(evt);
      }
    }
  }

  /**
   * Get raw pointer to buffer.
   */
  T* data() const {
    return buf;
  }

private:
  /**
   * Buffer.
   */
  T* buf;

  /**
   * Event to record after operation.
   */
  void* evt;
};
}
